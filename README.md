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

## Topics

| Folder | Concept |
|---|---|
| `bernoulli/` | Bernoulli MLE three ways: closed form, hand-derived SGD, PyTorch SGD — with a proof-by-computation that autograd reproduces the hand gradient exactly |
