import torch

EPS = 1e-8


class BCELossMaskOutput(torch.nn.BCELoss):
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "none",
        silent_weight=1,
    ):
        super(BCELossMaskOutput, self).__init__(
            size_average, reduce, reduction=reduction,
        )
        self.silent_weight = silent_weight

    def forward(self, scores, labels, mask=None):
        if mask is None:
            mask = torch.ones_like(labels)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        loss = super(BCELossMaskOutput, self).forward(scores, labels)
        loss[labels == 0] *= self.silent_weight
        loss = loss.reshape(mask.shape)
        loss = (loss * mask).sum(-1) / mask.sum(-1)
        loss = loss.mean()
        return loss

