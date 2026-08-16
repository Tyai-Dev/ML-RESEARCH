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


class MLP(nn.Module):
    """784 -> 512 -> 256 -> 128 -> 10, ReLU between layers (~550k
    params). ReLU is still the standard for plain MLPs; GELU is the
    modern smooth variant (what the GPT uses) — swap the activation
    below to try it. Each hidden layer lets the model compose
    features instead of matching one rigid template per class — the
    exact failure the linear floor exposed on cursive KMNIST."""

    def __init__(self, d_in: int = 784, n_classes: int = 10,
                 hidden=(512, 256, 128), dropout: float = 0.2,
                 batchnorm: bool = True):
        super().__init__()
        dims = [d_in, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(a, b))
            if batchnorm:                    # unit-scale activations:
                layers.append(nn.BatchNorm1d(b))   # per-batch, learned
            layers.append(nn.ReLU())
            if dropout:                      # regularize: random zeros
                layers.append(nn.Dropout(dropout))  # (train mode only)
        layers.append(nn.Linear(dims[-1], n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """The convolutional member — built for exactly the failure the
    CIFAR MLP exposed (tr-acc 83% / val-acc 59%): fully-connected
    layers give every pixel position its own weights, so a cat shifted
    three pixels is a new cat. Convolution SHARES weights across
    positions (one cat-ear detector, applied everywhere), sees local
    patches (shape lives locally), and pooling buys translation
    tolerance.

    Three stages of [Conv-BN-ReLU, Conv-BN-ReLU, MaxPool], widening
    64 -> 128 -> 256 channels while halving resolution, then global
    average pooling and a linear head (~1.2M params). Input arrives
    flat (the pipeline contract); the forward reshapes to (C, H, W).
    Works for (3,32,32) CIFAR and (1,28,28) MNIST-likes alike."""

    def __init__(self, in_shape=(3, 32, 32), n_classes: int = 10,
                 dropout: float = 0.3):
        super().__init__()
        self.in_shape = in_shape

        def block(c_in, c_out):
            return [nn.Conv2d(c_in, c_out, 3, padding=1),
                    nn.BatchNorm2d(c_out), nn.ReLU()]

        chans = [in_shape[0], 64, 128, 256]
        stages = []
        for a, b in zip(chans[:-1], chans[1:]):
            stages += block(a, b) + block(b, b) + [nn.MaxPool2d(2)]
        self.features = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Flatten(),
                                  nn.Dropout(dropout),
                                  nn.Linear(chans[-1], n_classes))

    def forward(self, x):                    # (B, C*H*W) flat
        x = x.view(-1, *self.in_shape)
        return self.head(self.features(x))


def describe(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters())
    return f"{type(model).__name__} ({n:,} params)"
