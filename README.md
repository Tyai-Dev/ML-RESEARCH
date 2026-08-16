# ML-RESEARCH

Personal ML research, done by hand. One folder per concept, each
self-contained: a runnable, heavily documented `.py` and a `.tex` deriving
the math. No framework — just VS Code, the `ml-research` conda env
(numpy, matplotlib, torch), and MiKTeX.

## Workflow

- **F5** runs the file you're editing (launch config "current file";
  breakpoints work).
- **Ctrl+Shift+B** compiles the `.tex` you're editing with pdflatex
  (LaTeX Workshop's save-to-compile works too).

Conventions: every script *verifies* its math (asserts against theory);
every hand-derived gradient is checked against autograd by driving both
through identical schedules; final results in tex get their own numbered
equation.

**Results site**: `python site/build.py` generates a static HTML site in
`site/out/` — one page per experiment with tabs for the verification
checklist (the assert-backed ": OK" lines), animations (mp4/scrubbable
player), plots, the compiled LaTeX PDF, the captured terminal output,
and the source. No server; open `site/out/index.html` or
`python -m http.server -d site/out`. A page only builds if every assert
in its script passes.

## Experiments

| Folder | Concept |
|---|---|
| `bernoulli/` | Bernoulli MLE in three files — `mle_theoretical.py` (closed form vs brute force), `mle_practical_pure.py` (GD + SGD by hand, with an *animation* of the estimator descending the NLL), `mle_practical_pytorch.py` (autograd trajectory == hand trajectory, exactly) |
| `gaussian/` | 1D normal MLE for (mu, sigma): same four routes; the MLE variance is the biased 1/n estimator |
| `linear-regression/` | Normal equations vs GD/SGD, and why the closed form is not always the right tool (cost, conditioning — with a float32 demo where normal equations lose to QR by 4000x — streaming, generality) |
| `logistic-regression/` | The conditional-Bernoulli view of classification: Y\|X=x ~ Bernoulli(p(x)), linear log-odds, conditional MLE. One solver per file over a shared `common.py` — SGD (logged noise ball + autograd identity), GD (geometric ‖grad‖ decay; separable divergence), Newton/IRLS (‖grad‖ squares away in 8 rows) — each ending in a sklearn classification report; the SGD film in `logistic_animation.py`; and `logistic_mnist.py`: the same unchanged loops on real images (MNIST 3 vs 5, 96.7%, ŵ rendered as the template it learned, damped Newton after the singular-Hessian crash) |
| `poisson/` | Count data: Y\|x ~ Poisson(exp(w·x)); gradient (λ−y)x; Newton/IRLS — and the GLM unification theorem: canonical link ⇒ gradient = (mean − y)·x for the whole family |
| `softmax-regression/` | Bernoulli → multinoulli: Y\|x ~ Multinoulli(softmax(Wx)); cross-entropy gradient (softmax − onehot)⊗x; identifiability up to row-shift; multiclass Bayes floor E[1 − max p] met |
| `multi-output-regression/` | Linear → multivariate Gaussian: vector response, Frobenius loss separates per column so the closed form survives; SUR collapse; the one-hot "linearization trick" and its broken probabilities |
| `perceptron/` | Distribution-free: online mistakes, no p(x). Novikoff's (R/γ)² bound proved and verified; multiclass promote/demote; noise breaks convergence forever |
| `passive-aggressive/` | Online learning as per-step optimization: min ‖w−w_t‖² s.t. hinge = 0, closed form via KKT (verified vs brute force); PA-I/PA-II noise robustness raced; multiclass PA |
| `svm/` | Margin maximization: hard margin from geometry, KKT ⇒ support vectors (dropping 1324 non-SVs moves w by 0.0000; dropping 176 SVs by 0.25), soft margin = hinge + L2, Pegasos, logistic comparison, kernel remark |

## MNIST classifier (real images, torch, model zoo)

| File | Role |
|---|---|
| `mnist-classifier/models.py` | The zoo, smallest first: `SoftmaxRegression` (784→10 linear, 7.9k params) — the floor every later model must beat on the same splits |
| `mnist-classifier/training.py` | The generic engine: `fit(model, loss_fn, ...)` — model and loss are arguments; quarter-epoch running stats, per-epoch val + best-checkpoint restore, seeded batches |
| `mnist-classifier/mnist_softmax.py` | The pipeline: LOAD (55k/5k/10k) → LOOK (examples + balance) → TRAIN → EXAMINE (10-class sklearn report, confusion matrix + worst pairs, confidence/calibration, the 10 learned templates as images, most confident mistakes). Linear floor: **92.74%** test |

## LLM track

Building an LLM by hand in raw PyTorch (design:
`docs/plans/2026-08-14-llm-track-design.md`). One corpus (Tiny
Shakespeare, char-level), one measuring stick (validation NLL /
perplexity on the same held-out slice), and a ladder where each rung
must numerically beat the last:

| Rung | Model | val NLL (nats/char) | PPL |
|---|---|---|---|
| `llm/bigram/` | Conditional multinoulli: counts = MLE = GD = SGD(Polyak) = autograd, all verified equal | 2.4819 (Laplace α=1) | **11.96** |
| `llm/ngram-mlp/` | Counting dies (U-turn measured: k=3 sweet spot, k=5 worse than bigram) → embeddings + tanh MLP (Bengio 2003), backprop by hand == autograd to 1e-16 | 1.7583 | **5.80** |
| `llm/attention/` | Attention derived + hand forward/backward == autograd to 1e-16 (causality proven by perturbation); one transformer block, T=64 | 1.6254 | **5.08** |
| `llm/gpt/` | Full decoder assembled from the verified parts: 4 blocks, T=128, 3.21M params, weight tying, AdamW + warmup/cosine + clipping (each ablated — verdict: insurance, not magic); 2.3 min on the 4070 | 1.4724 | **4.36** |

**Talk to them**: `llm/chat/chat.py` (F5) is one REPL over every model — pick a rung with `/model`, type a line, and it becomes part of a play the model continues autoregressively. Models build on first selection and cache to `llm/chat/checkpoints/`; switching rungs mid-conversation lets you *feel* the ladder (the bigram babbles no matter what you say; the GPT answers in blank verse; `gpt-austen` switches to drawing-room prose).

Beyond the ladder (same measuring stick — NLL per **char**, so tokenizers stay comparable):

| Folder | Experiment | Headline number |
|---|---|---|
| `llm/tokenization/` | BPE by hand (lossless roundtrip asserted; vocab rediscovers ' the', '\n\n', 'ICHARD'); rung-4 GPT on 515 tokens overfits on schedule → early stopping enters | chars win: 1.5165 vs 1.4724 — data, not context, is the bottleneck |
| `llm/finetuning/` | Shakespeare checkpoint → Austen: zero-shot 1.7341, finetune (lr 1e-4) 1.0978, identical-budget scratch 1.2393 | pretraining worth 0.14 nats/char |
| `llm/scaling/` | Param sweep (0.2M–8M) vs data sweep (10%–100%) under a fixed budget | param gains stall + overfit gap grows; every data doubling still pays — data-limited regime, measured |

## Theory (reference pages)

| Folder | Contents |
|---|---|
| `Theory/distributions/` | Bernoulli, multinoulli, normal, multivariate normal: densities, moments, MLEs, sufficient statistics — every moment formula verified by sampling |
| `Theory/optimizers/` | Newton–Raphson, GD, heavy ball, Nesterov, BFGS/L-BFGS, AdaGrad, RMSProp, Adam(W): update rules + all implemented and raced on Rosenbrock |
| `Theory/learning-theory/` | Two documents: `learning-theory.tex` — LLN, Hoeffding, Azuma, Glivenko–Cantelli & GC classes; `learnability.tex` — self-contained Learnability à la Shalev-Shwartz & Ben-David (UML Part I): formal model, ERM & overfitting, PAC with full finite-class proofs, uniform convergence, No-Free-Lunch, VC dimension, Sauer, the Fundamental Theorem — with a demo measuring the 1/m and 1/√m rates and a margin-condition surprise |
| `Theory/statistics/` | Estimators, bias/variance, MSE decomposition, sufficiency & factorization, Fisher information, Cramér–Rao (Bernoulli MLE shown efficient), MLE asymptotics |
| `Theory/markov-chains/` | The probability behind n-grams: Markov property, k-th order → first-order reduction (verified exactly on Shakespeare trigrams), Chapman–Kolmogorov, stationarity, mixing rate = \|λ₂\| (measured to 4 decimals), ergodic theorem — plus the library's first MLE on **dependent** data: √T rate, occupancy-weighted variance p(1−p)/(Tπᵢ) (matches to 2.7%), 94.7% CI coverage, π̂₀ from many paths, all on a weather chain with nothing linguistic about it |
| `Theory/optimization/` | When does grad = 0 solve a problem: stationarity, saddles, convexity (strong/smooth/bounded), Lagrange multipliers, KKT conditions |
| `Theory/information-theory/` | Entropy, KL & Gibbs' inequality, cross-entropy = entropy + KL, MLE = KL minimization (verified to 1e-12), mutual information |
| `Theory/losses/` | The loss catalogue: 0-1 and its convex surrogates (hinge, logistic, exponential, squared), calibration — what each minimizer remembers (logit(p) / 2p−1 / sign, verified numerically), regression losses and robustness (mean vs median vs Huber under outliers), the choosing table |
| `Theory/evaluation/` | Measuring success: confusion matrix → accuracy/precision/recall/F1 (accuracy's lie under imbalance measured: 0.973 with recall 0), ROC with AUC = P(s⁺>s⁻) proved and verified to 3e-4, PR collapse under imbalance (AP 0.86→0.45 while AUC moves 0.003), calibration → cost-optimal threshold τ* = c_FP/(c_FP+c_FN) (theory 0.167, measured 0.161), the choosing table — plus a threshold-sweep animation: the decision line sliding while dots travel the ROC/PR curves |
