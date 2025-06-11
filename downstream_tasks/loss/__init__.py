import torch


class NormalizingKLDivLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.loss_fn = torch.nn.KLDivLoss(*args, **kwargs)

    def forward(self, logits, target):
        #inp = logits - logits.min(dim=1)[0].unsqueeze(-1) + 1e-5
        inp = torch.nn.functional.sigmoid(logits)
        inp = inp / inp.sum(dim=1).unsqueeze(-1)
        inp = inp.log()
        return self.loss_fn(inp, target)
