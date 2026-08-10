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

## Experiments

| Folder | Concept |
|---|---|
| `bernoulli/` | Bernoulli MLE: closed form, GD, SGD, and SGD via autograd — trajectories of hand vs autograd coincide exactly |
| `gaussian/` | 1D normal MLE for (mu, sigma): same four routes; the MLE variance is the biased 1/n estimator |
| `linear-regression/` | Normal equations vs GD/SGD, and why the closed form is not always the right tool (cost, conditioning — with a float32 demo where normal equations lose to QR by 4000x — streaming, generality) |
| `logistic-regression/` | The conditional-Bernoulli view of classification: Y\|X=x ~ Bernoulli(p(x)), linear log-odds, conditional MLE. First model with no closed form (transcendental stationarity) — solved by Newton/IRLS in 8 steps vs GD's 3000; separable-data MLE nonexistence; fitted classifier hits the Bayes floor to 3 decimals |

## Theory (reference pages)

| Folder | Contents |
|---|---|
| `Theory/distributions/` | Bernoulli, multinoulli, normal, multivariate normal: densities, moments, MLEs, sufficient statistics — every moment formula verified by sampling |
| `Theory/optimizers/` | Newton–Raphson, GD, heavy ball, Nesterov, BFGS/L-BFGS, AdaGrad, RMSProp, Adam(W): update rules + all implemented and raced on Rosenbrock |
| `Theory/learning-theory/` | Two documents: `learning-theory.tex` — LLN, Hoeffding, Azuma, Glivenko–Cantelli & GC classes; `learnability.tex` — self-contained Learnability à la Shalev-Shwartz & Ben-David (UML Part I): formal model, ERM & overfitting, PAC with full finite-class proofs, uniform convergence, No-Free-Lunch, VC dimension, Sauer, the Fundamental Theorem — with a demo measuring the 1/m and 1/√m rates and a margin-condition surprise |
| `Theory/statistics/` | Estimators, bias/variance, MSE decomposition, sufficiency & factorization, Fisher information, Cramér–Rao (Bernoulli MLE shown efficient), MLE asymptotics |
| `Theory/optimization/` | When does grad = 0 solve a problem: stationarity, saddles, convexity (strong/smooth/bounded), Lagrange multipliers, KKT conditions |
| `Theory/information-theory/` | Entropy, KL & Gibbs' inequality, cross-entropy = entropy + KL, MLE = KL minimization (verified to 1e-12), mutual information |
