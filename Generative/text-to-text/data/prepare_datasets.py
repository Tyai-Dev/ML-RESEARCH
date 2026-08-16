r"""The dataset pantry — one corpus per capability we want to teach.

The ladder's lesson so far: a language model is a continuation engine
for whatever distribution it was trained on. Shakespeare in, blank
verse out. So "what can the model do" is decided HERE, in the data,
before a single parameter exists. This script stocks the pantry:

  pretrain   tinystories.txt           general English. TinyStories V2
                                       (GPT-4-written children's
                                       stories, arXiv:2305.07759) — the
                                       canonical proof that models our
                                       size can speak fluent English if
                                       the distribution is simple.
  chat       soda_chat.txt             everyday dialogues (allenai/soda,
                                       arXiv:2212.10465) reformatted as
                                       User:/Bot: turns — finetune on
                                       this and the REPL's play-trick
                                       becomes the actual training
                                       distribution.
  instruct   tinystories_instruct.txt  command -> story records
                                       (roneneldan/TinyStoriesInstruct):
                                       the model learns to FOLLOW a
                                       specification (features, words
                                       to use, a summary) instead of
                                       merely continuing.
  summarize  tinystories_summarize.txt Story: -> Summary: pairs, parsed
                                       out of the instruct records —
                                       the inverse task, for free.

All four are cleaned to one compact ASCII alphabet (quotes and dashes
normalized, everything else dropped) so every rung keeps a small
character vocabulary — the same tokenizer-lock-in lesson as chat.py.

Downloads are surgical: the sources total ~5.5GB but we take byte
ranges (text) or leading row groups (parquet), so ~100MB moves over
the wire for the default sizes.

Run me with F5, or:
  python prepare_datasets.py                    # everything, defaults
  python prepare_datasets.py chat --mb 40       # one dataset, custom size
"""

import argparse
import os
import re
import sys

import requests

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
EOT = "<|endoftext|>"

TINYSTORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStoriesV2-GPT4-train.txt"
)
INSTRUCT_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/resolve/main/"
    "TinyStories-Instruct-train.txt"
)
SODA_URL = "https://huggingface.co/datasets/allenai/soda/resolve/main/train.parquet"

# Unicode that appears in these corpora, mapped into the ASCII alphabet.
_REPLACEMENTS = {
    "‘": "'", "’": "'", "‚": "'", "′": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-",
    "…": "...",
    " ": " ",
    "\r": "",
}


def clean(text: str) -> str:
    """Normalize punctuation, drop what is left outside ASCII, tidy
    whitespace. The result defines the character vocabulary, so less
    is more."""
    for bad, good in _REPLACEMENTS.items():
        text = text.replace(bad, good)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[ \t]+\n", "\n", text)          # trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)          # runs of blank lines
    return text


_fetch_cache = {}


def fetch_head(url: str, n_bytes: int) -> str:
    """First n_bytes of a remote text file via an HTTP Range request,
    cut back to the last complete <|endoftext|> record."""
    key = (url, n_bytes)
    if key not in _fetch_cache:
        name = url.rsplit("/", 1)[-1]
        print(f"  fetching first {n_bytes // 2**20}MB of {name} ...")
        r = requests.get(url, headers={"Range": f"bytes=0-{n_bytes - 1}"}, timeout=120)
        r.raise_for_status()
        text = r.content.decode("utf-8", errors="ignore")
        _fetch_cache[key] = text[: text.rfind(EOT)]
    return _fetch_cache[key]


def records_of(text: str):
    """Split a TinyStories-family file into its <|endoftext|>-separated
    records."""
    return [r.strip() for r in text.split(EOT) if r.strip()]


def save(name: str, blocks, label: str) -> str:
    """Join records with blank lines (our char-level record separator),
    write, and report. Returns the path."""
    out = clean("\n\n".join(blocks)) + "\n"
    path = os.path.join(DATA_DIR, name)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(out)
    print(
        f"  {name}: {len(out) / 2**20:.1f}MB, {len(blocks)} {label}, "
        f"alphabet {len(set(out))} chars"
    )
    return path


# ----------------------------------------------------------------------
# pretrain: TinyStories V2, plain stories
# ----------------------------------------------------------------------
def prep_pretrain(mb: int):
    # stories average ~900 chars; fetch a hair extra, trim to size
    text = fetch_head(TINYSTORIES_URL, mb * 2**20 + 2**16)
    stories, total = [], 0
    for rec in records_of(text):
        stories.append(rec)
        total += len(rec) + 2
        if total >= mb * 2**20:
            break
    save("tinystories.txt", stories, "stories")


# ----------------------------------------------------------------------
# instruct: specification -> story records, kept in source format
# ----------------------------------------------------------------------
def prep_instruct(mb: int):
    text = fetch_head(INSTRUCT_URL, mb * 2**20 + 2**16)
    recs, total = [], 0
    for rec in records_of(text):
        recs.append(rec)
        total += len(rec) + 2
        if total >= mb * 2**20:
            break
    save("tinystories_instruct.txt", recs, "records")


# ----------------------------------------------------------------------
# summarize: Story -> Summary pairs parsed from the instruct records
# ----------------------------------------------------------------------
_FIELD = re.compile(r"^(Features|Words|Summary|Story|Random sentence):", re.M)


def parse_fields(rec: str) -> dict:
    """An instruct record is 'Key: value' sections; Story's value runs
    to the end of the record."""
    fields = {}
    matches = list(_FIELD.finditer(rec))
    for m, nxt in zip(matches, matches[1:] + [None]):
        end = nxt.start() if nxt else len(rec)
        fields[m.group(1)] = rec[m.end():end].strip()
    return fields


def prep_summarize(mb: int, source_mb: int):
    # summaries are ~10% of a record; read extra instruct text
    text = fetch_head(INSTRUCT_URL, source_mb * 2**20 + 2**16)
    pairs, total = [], 0
    for rec in records_of(text):
        f = parse_fields(rec)
        if "Story" in f and "Summary" in f and f["Story"] and f["Summary"]:
            block = f"Story: {f['Story']}\nSummary: {f['Summary']}"
            pairs.append(block)
            total += len(block) + 2
            if total >= mb * 2**20:
                break
    save("tinystories_summarize.txt", pairs, "story/summary pairs")


# ----------------------------------------------------------------------
# chat: SODA dialogues -> User:/Bot: transcript format
# ----------------------------------------------------------------------
def prep_chat(mb: int):
    import fsspec
    import pyarrow.parquet as pq

    print("  reading leading row groups of soda/train.parquet ...")
    dialogues, total = [], 0
    with fsspec.open(SODA_URL, "rb") as f:
        pf = pq.ParquetFile(f)
        for gi in range(pf.num_row_groups):
            tbl = pf.read_row_group(gi, columns=["dialogue", "speakers"])
            for row in tbl.to_pylist():
                turns, speakers = row["dialogue"], row["speakers"]
                # two-speaker, strictly alternating dialogues only:
                # the User/Bot mapping must be unambiguous
                names = list(dict.fromkeys(speakers))
                if len(names) != 2 or len(turns) < 2:
                    continue
                if any(s == t for s, t in zip(speakers, speakers[1:])):
                    continue
                role = {names[0]: "User", names[1]: "Bot"}
                block = "\n".join(
                    f"{role[s]}: {t.strip()}" for s, t in zip(speakers, turns)
                )
                dialogues.append(block)
                total += len(block) + 2
                if total >= mb * 2**20:
                    break
            if total >= mb * 2**20:
                break
    save("soda_chat.txt", dialogues, "dialogues")


# ----------------------------------------------------------------------
DATASETS = {
    "pretrain": lambda a: prep_pretrain(a.mb or 50),
    "chat": lambda a: prep_chat(a.mb or 25),
    "instruct": lambda a: prep_instruct(a.mb or 25),
    "summarize": lambda a: prep_summarize(a.mb or 15, source_mb=(a.mb or 15) * 8),
}


OUTPUTS = [
    "tinystories.txt", "soda_chat.txt",
    "tinystories_instruct.txt", "tinystories_summarize.txt",
]


def write_alphabet():
    """One canonical character vocabulary for the whole pantry: the
    union over every prepared file. A model pretrained on stories must
    already own an embedding for every char the chat/instruct finetunes
    will feed it — deriving the vocab per-corpus would break that."""
    chars = set()
    for name in OUTPUTS:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            with open(path, encoding="ascii") as f:
                chars |= set(f.read())
    path = os.path.join(DATA_DIR, "alphabet.txt")
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("".join(sorted(chars)))
    print(f"  alphabet.txt: {len(chars)} chars (union of prepared files)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "which", nargs="*", choices=list(DATASETS),
        help="datasets to prepare (default: all)",
    )
    ap.add_argument("--mb", type=int, help="target size in MB (per dataset)")
    args = ap.parse_args()
    for name in args.which or list(DATASETS):
        print(f"[{name}]")
        DATASETS[name](args)
    write_alphabet()
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
