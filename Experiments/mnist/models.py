r"""The model zoo for the MNIST classifier — smallest member first.

Every model here is a torch nn.Module mapping a batch of flattened
images (B, 784) to a batch of LOGITS (B, 10). The softmax that turns
logits into probabilities lives inside the loss (CrossEntropyLoss
computes log-softmax internally via the log-sum-exp trick — composing
them separately is both slower and numerically worse; the same reason
rung 1 of the LLM track used cross_entropy on logits).

SoftmaxRegression is exactly Algorithms/classification/multiclass/softmax/softmax.py — the
conditional multinoulli Y|x ~ Multinoulli(softmax(Wx)), one weight
vector per class — wearing torch clothes. Its 10 weight rows are 10
IMAGES: the templates the classifier matches against, visualized by
the experiment. It is the floor every later model must beat; the zoo
grows above it (hidden layers, convolutions) as we work on making it
better.
"""

import torch.nn as nn


class SoftmaxRegression(nn.Module):
    """784 -> 10, one linear layer. ~7.9k parameters."""

    def __init__(self, d_in: int = 784, n_classes: int = 10):
        super().__init__()
        self.linear = nn.Linear(d_in, n_classes)

    def forward(self, x):                    # (B, 784) -> (B, 10) logits
        return self.linear(x)

    def templates(self):
        """The weight matrix as n_classes images (28, 28) — what each
        class's neuron looks for."""
        return self.linear.weight.detach().cpu().numpy().reshape(
            -1, 28, 28)


def describe(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters())
    return f"{type(model).__name__} ({n:,} params)"
