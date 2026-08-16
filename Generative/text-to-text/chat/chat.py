r"""Chat with the ladder — one REPL, every model, autoregression live.

"Chatting" with a language model IS the ladder's math running in a
loop: the model predicts a distribution over the next token, we sample
it, append it to the context, and FEED THE CONTEXT BACK IN — the same
autoregressive factorization P(c_1..c_T) = prod P(c_i | c_<i) that
rung 1 derived, executed one step at a time. Nothing else. A "chat
model" is this loop plus a conversation transcript as the context.

None of these models is instruction-tuned: they are Shakespeare (or
Austen) CONTINUATION engines. The trick that makes chatting fun: your
messages become lines in a play the model believes it is writing, so
it answers in character. And because every model shares one alphabet
and one interface, you can switch rungs mid-conversation and FEEL the
ladder: the bigram babbles letter-texture regardless of what you say
(context: 1 char), the MLP echoes word-shapes (context: 8), the
transformer models actually stay with your words (context: 64-128).

Models are built on first selection and cached to chat/checkpoints/
(the Shakespeare GPT reuses Generative/text-to-text/gpt/gpt_checkpoint.pt if you already
ran gpt_train.py; the Austen model finetunes from it; the BPE model
trains to its early-stopping region — see each folder's tex).

Commands:
  /models           list models, status, and scores
  /model <n>        switch model (builds + caches on first use)
  /temp <x>         sampling temperature (default 0.9; lower = safer)
  /topk <k|off>     top-k filter (default 40)
  /len <n>          max reply length in chars (default 300)
  /reset            clear the conversation transcript
  /quit             leave

Run me with F5.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]  # Generative/text-to-text/
sys.path.insert(0, str(ROOT))
for sub in ("bigram", "ngram-mlp", "attention", "gpt", "tokenization", "finetuning"):
    sys.path.insert(0, str(ROOT / sub))

from common import SEED, get_device, load_everything, decode  # noqa: E402
import bigram_practical_pure as bigrammod  # noqa: E402
import mlp_practical_pure as mlppure  # noqa: E402
import mlp_practical_pytorch as mlpmod  # noqa: E402
import attention_practical_pytorch as attnmod  # noqa: E402
import gpt_model as gptm  # noqa: E402
import gpt_train as gptt  # noqa: E402
import bpe_theoretical as bpemod  # noqa: E402
import finetune_practical_pytorch as ftmod  # noqa: E402

CKPT_DIR = Path(__file__).with_name("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)
GPT_CKPT = ROOT / "gpt" / "gpt_checkpoint.pt"

DEVICE = get_device()
TRAIN_IDS, VAL_IDS, CHARS, STOI, ITOS = load_everything()
V = len(CHARS)


def sanitize(text: str) -> str:
    """Every model speaks the 65-char Shakespeare alphabet; drop what
    it cannot represent (the tokenizer lock-in lesson)."""
    return "".join(c for c in text if c in STOI)


def pick_next(logits: torch.Tensor, temperature, top_k, g) -> int:
    """One autoregressive step: logits -> distribution -> sample."""
    logits = logits / max(temperature, 1e-3)
    if top_k:
        cut = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < cut, float("-inf"))
    p = torch.softmax(logits, dim=0).cpu()
    return int(torch.multinomial(p, 1, generator=g))


def stop_here(reply: str, min_chars=40) -> bool:
    """End the reply at a blank line (a full speech), once it has said
    something."""
    return len(reply) > min_chars and "\n\n" in reply[min_chars:]


# ----------------------------------------------------------------------
# Wrappers: one .reply() signature over very different machines
# ----------------------------------------------------------------------
class CountBigram:
    """Rung 1: P(next|current) as the count table. Context: 1 char."""

    def __init__(self):
        _, self.P, _ = bigrammod.count_model(TRAIN_IDS, V)

    def reply(self, history, max_chars, temperature, top_k, g):
        a = STOI.get(history[-1], 0) if history else 0
        out = ""
        for _ in range(max_chars):
            logits = torch.log(torch.tensor(self.P[a]) + 1e-12)
            a = pick_next(logits, temperature, top_k, g)
            out += ITOS[a]
            if stop_here(out):
                break
        return out


class MLPWrap:
    """Rung 2: the neural n-gram. Context: 8 chars."""

    def __init__(self, model):
        self.model = model.eval()

    @torch.no_grad()
    def reply(self, history, max_chars, temperature, top_k, g):
        k = mlppure.CONTEXT_K
        ctx = [STOI.get(c, 0) for c in history[-k:]]
        ctx = [0] * (k - len(ctx)) + ctx
        out = ""
        for _ in range(max_chars):
            x = torch.tensor([ctx], device=DEVICE)
            c = pick_next(self.model(x)[0], temperature, top_k, g)
            out += ITOS[c]
            ctx = ctx[1:] + [c]
            if stop_here(out):
                break
        return out


class BlockLM:
    """Rungs 3-4 (+ Austen): a transformer fed its own output.
    Context: `block` chars of the transcript."""

    def __init__(self, model, block):
        self.model = model.eval()
        self.block = block

    @torch.no_grad()
    def reply(self, history, max_chars, temperature, top_k, g):
        ids = [STOI[c] for c in history[-self.block :]] or [0]
        out = ""
        for _ in range(max_chars):
            x = torch.tensor([ids[-self.block :]], device=DEVICE)
            c = pick_next(self.model(x)[0, -1], temperature, top_k, g)
            out += ITOS[c]
            ids.append(c)
            if stop_here(out):
                break
        return out


class BPEWrap:
    """The BPE-GPT: same loop, bigger alphabet — whole words per step."""

    def __init__(self, model, merges, vocab):
        self.model = model.eval()
        self.stoi = {t: i for i, t in enumerate(vocab)}
        self.vocab = vocab
        self.merges = merges
        self.block = self.model.cfg.block_size

    @torch.no_grad()
    def reply(self, history, max_chars, temperature, top_k, g):
        toks = bpemod.encode(history[-400:], self.merges, {})
        ids = [self.stoi[t] for t in toks][-self.block :] or [0]
        out = ""
        while len(out) < max_chars:
            x = torch.tensor([ids[-self.block :]], device=DEVICE)
            t = pick_next(self.model(x)[0, -1], temperature, top_k, g)
            out += self.vocab[t]
            ids.append(t)
            if stop_here(out):
                break
        return out


# ----------------------------------------------------------------------
# Builders: train on first use, cache forever
# ----------------------------------------------------------------------
def _train_lm(model, get_xy, steps_schedule, label, clip=None):
    """The shared loop: sample batch, cross-entropy, step."""
    g = torch.Generator().manual_seed(SEED)
    t0, done = time.perf_counter(), 0
    for lr, n_steps in steps_schedule:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(n_steps):
            x, y = get_xy(g)
            logits = model(x)  # (B,V) or (B,T,V)
            loss = F.cross_entropy(logits.reshape(y.numel(), -1), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            done += 1
    print(f"  [{label}] trained {done} steps " f"in {time.perf_counter() - t0:.0f}s")
    return model


def build_bigram():
    return CountBigram()


def build_mlp():
    path = CKPT_DIR / "mlp.pt"
    torch.manual_seed(SEED)
    model = mlpmod.NGramMLP(V).to(DEVICE)
    if path.exists():
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        Xtr, Ytr = mlpmod.windows_tensor(TRAIN_IDS, DEVICE)

        def get_xy(g):
            rows = torch.randint(0, len(Xtr), (mlpmod.BATCH,), generator=g).to(DEVICE)
            return Xtr[rows], Ytr[rows]

        _train_lm(model, get_xy, mlpmod.LR_SCHEDULE, "ngram-mlp")
        torch.save(model.state_dict(), path)
    return MLPWrap(model)


def build_attention():
    path = CKPT_DIR / "attention.pt"
    torch.manual_seed(SEED)
    model = attnmod.OneBlockLM(V).to(DEVICE)
    if path.exists():
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        tr = torch.from_numpy(TRAIN_IDS.copy())

        def get_xy(g):
            x, y = attnmod.get_batch(tr, attnmod.BATCH, g)
            return x.to(DEVICE), y.to(DEVICE)

        _train_lm(model, get_xy, attnmod.LR_SCHEDULE, "attention")
        torch.save(model.state_dict(), path)
    return BlockLM(model, attnmod.CONTEXT_T)


def _load_or_train_gpt():
    torch.manual_seed(SEED)
    if GPT_CKPT.exists():
        saved = torch.load(GPT_CKPT, weights_only=False)
        cfg = gptm.GPTConfig(**saved["config"])
        model = gptm.GPT(cfg).to(DEVICE)
        model.load_state_dict(saved["state_dict"])
        return model, cfg
    print("  no gpt_checkpoint.pt — training rung 4 now (~2.5 min)")
    cfg = gptm.GPTConfig(vocab_size=V)
    model = gptm.GPT(cfg).to(DEVICE)
    tr = torch.from_numpy(TRAIN_IDS.copy())
    opt = gptt.configure_adamw(model)
    g = torch.Generator().manual_seed(SEED)
    for step in range(gptt.MAX_STEPS):
        for group in opt.param_groups:
            group["lr"] = gptt.lr_at(step)
        x, y = gptt.get_batch(tr, cfg.block_size, gptt.BATCH, g, DEVICE)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gptt.GRAD_CLIP)
        opt.step()
    va = torch.from_numpy(VAL_IDS.copy())
    torch.save(
        {
            "config": cfg.__dict__,
            "state_dict": model.state_dict(),
            "val_nll": gptt.full_nll(model, va, cfg.block_size, DEVICE),
        },
        GPT_CKPT,
    )
    return model, cfg


def build_gpt():
    model, cfg = _load_or_train_gpt()
    return BlockLM(model, cfg.block_size)


def build_austen():
    path = CKPT_DIR / "austen.pt"
    torch.manual_seed(SEED)
    if path.exists():
        saved = torch.load(path, weights_only=False)
        cfg = gptm.GPTConfig(**saved["config"])
        model = gptm.GPT(cfg).to(DEVICE)
        model.load_state_dict(saved["state_dict"])
    else:
        model, cfg = _load_or_train_gpt()
        text = ftmod.load_austen(set(CHARS))
        ids = np.array([STOI[c] for c in text], dtype=np.int64)
        tr = torch.from_numpy(ids[: int(len(ids) * 0.9)].copy())

        def get_xy(g):
            return gptt.get_batch(tr, cfg.block_size, 64, g, DEVICE)

        _train_lm(
            model, get_xy, [(ftmod.FT_LR, ftmod.BUDGET)], "austen finetune", clip=1.0
        )
        torch.save({"config": cfg.__dict__, "state_dict": model.state_dict()}, path)
    return BlockLM(model, cfg.block_size)


def build_bpe():
    path = CKPT_DIR / "bpe.pt"
    torch.manual_seed(SEED)
    if path.exists():
        saved = torch.load(path, weights_only=False)
        merges = [tuple(m) for m in saved["merges"]]
        vocab = saved["vocab"]
        cfg = gptm.GPTConfig(**saved["config"])
        model = gptm.GPT(cfg).to(DEVICE)
        model.load_state_dict(saved["state_dict"])
    else:
        print("  training BPE merges + GPT to its early-stop region " "(~2 min)")
        text = "".join(ITOS[i] for i in TRAIN_IDS)
        merges, _ = bpemod.train_bpe(text, bpemod.N_MERGES)
        vocab = sorted(set(text)) + [a + b for a, b in merges]
        stoi = {t: i for i, t in enumerate(vocab)}
        toks = torch.tensor([stoi[t] for t in bpemod.encode(text, merges, {})])
        cfg = gptm.GPTConfig(vocab_size=len(vocab))
        model = gptm.GPT(cfg).to(DEVICE)

        def get_xy(g):
            ix = torch.randint(len(toks) - cfg.block_size - 1, (64,), generator=g)
            x = torch.stack([toks[i : i + cfg.block_size] for i in ix]).to(DEVICE)
            y = torch.stack([toks[i + 1 : i + cfg.block_size + 1] for i in ix]).to(
                DEVICE
            )
            return x, y

        _train_lm(model, get_xy, [(6e-4, 1250)], "bpe-gpt", clip=1.0)
        torch.save(
            {
                "config": cfg.__dict__,
                "state_dict": model.state_dict(),
                "merges": merges,
                "vocab": vocab,
            },
            path,
        )
    return BPEWrap(model, merges, vocab)


REGISTRY = [
    dict(
        key="bigram",
        build=build_bigram,
        ctx="1 char",
        desc="rung 1 — count table (instant build)",
        nll="2.48",
    ),
    dict(
        key="ngram-mlp",
        build=build_mlp,
        ctx="8 chars",
        desc="rung 2 — Bengio MLP, 68k params (~20s build)",
        nll="1.76",
    ),
    dict(
        key="attention",
        build=build_attention,
        ctx="64 chars",
        desc="rung 3 — one transformer block, 223k (~70s build)",
        nll="1.63",
    ),
    dict(
        key="gpt",
        build=build_gpt,
        ctx="128 chars",
        desc="rung 4 — the GPT, 3.21M (uses gpt_checkpoint.pt)",
        nll="1.47",
    ),
    dict(
        key="gpt-austen",
        build=build_austen,
        ctx="128 chars",
        desc="rung 4 finetuned on Pride & Prejudice (~1 min build)",
        nll="1.10 (Austen)",
    ),
    dict(
        key="gpt-bpe",
        build=build_bpe,
        ctx="~300 chars",
        desc="rung 4 on 515 BPE tokens (~2 min build)",
        nll="1.52/char",
    ),
]


def list_models(cache):
    print("\n  #  model        val NLL/char  context    status")
    for i, m in enumerate(REGISTRY, 1):
        if m["key"] in cache:
            status = "loaded"
        elif (
            (CKPT_DIR / f"{m['key'].split('-')[-1]}.pt").exists()
            or m["key"] == "bigram"
            or (m["key"] == "gpt" and GPT_CKPT.exists())
        ):
            status = "cached on disk"
        else:
            status = "builds on first use"
        print(
            f"  {i}  {m['key']:<12} {m['nll']:<13} {m['ctx']:<10} "
            f"{status}\n     {m['desc']}"
        )
    print()


def main():
    print(__doc__.split("Commands:")[0])
    print(f"device: {DEVICE}\n")
    cache = {}
    current = None
    history = ""
    temp, topk, maxlen = 0.9, 40, 300

    def switch(idx):
        nonlocal current
        m = REGISTRY[idx]
        if m["key"] not in cache:
            print(f"  building {m['key']} ...")
            cache[m["key"]] = m["build"]()
        current = m
        print(f"  now chatting with: {m['key']}  ({m['desc']})")

    list_models(cache)
    switch(3 if GPT_CKPT.exists() else 0)  # best available default

    while True:
        try:
            user = input(f"\n[{current['key']} t={temp} k={topk}] you> ")
        except EOFError:
            break
        user = user.strip().lstrip("﻿")  # BOM guard (piped stdin)
        if not user:
            continue
        if user.startswith("/"):
            cmd, *arg = user.split()
            if cmd == "/quit":
                break
            elif cmd == "/models":
                list_models(cache)
            elif cmd == "/model" and arg:
                try:
                    switch(int(arg[0]) - 1)
                except (ValueError, IndexError):
                    keys = [m["key"] for m in REGISTRY]
                    if arg[0] in keys:
                        switch(keys.index(arg[0]))
                    else:
                        print(f"  pick 1-{len(REGISTRY)} or a name")
            elif cmd == "/temp" and arg:
                temp = float(arg[0])
            elif cmd == "/topk" and arg:
                topk = None if arg[0] == "off" else int(arg[0])
            elif cmd == "/len" and arg:
                maxlen = int(arg[0])
            elif cmd == "/reset":
                history = ""
                print("  transcript cleared")
            else:
                print("  /models /model /temp /topk /len /reset /quit")
            continue

        clean = sanitize(user)
        if clean != user:
            print(
                "  (some characters aren't in the 65-char alphabet " "and were dropped)"
            )
        # your line enters the play; the model continues it
        history += (
            ("" if not history or history.endswith("\n") else "\n") + clean + "\n"
        )
        g = torch.Generator().manual_seed(
            torch.seed() % (2**31)
        )  # fresh sample each turn
        t0 = time.perf_counter()
        reply = cache[current["key"]].reply(history, maxlen, temp, topk, g)
        cut = reply.find("\n\n", 40)  # end at the speech's
        if cut != -1:  # blank line, dropping
            reply = reply[: cut + 1]  # the next speaker's stub
        history += reply
        print(f"\n{reply.strip()}")
        print(
            f"  -- {current['key']}, {len(reply)} chars in "
            f"{time.perf_counter() - t0:.1f}s"
        )


if __name__ == "__main__":
    main()
