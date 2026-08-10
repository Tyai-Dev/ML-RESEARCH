# Manual

Hand-worked concepts, one folder each. Every folder is self-contained:
a `.py` you can run directly and a `.tex` that explains the math —
no registries, no configs, no framework. When something here matures,
promote it into `src/mlr` + `research/`.

Workflow (VS Code):

- **F5** runs the file you're editing (pick the "current file" launch config
  once in the Run dropdown; it sticks).
- **Ctrl+Shift+B** compiles the `.tex` you're editing with pdflatex
  (LaTeX Workshop's auto-compile-on-save works too).

```
Manual/
└── bernoulli/
    ├── mle.py         # MLE two ways: closed form vs SGD on the NLL
    └── bernoulli.tex  # the math behind it
```
