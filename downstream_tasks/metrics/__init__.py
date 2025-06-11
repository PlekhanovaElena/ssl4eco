import torch
import torchmetrics


class UnnormalizedMetric(torchmetrics.Metric):
    """Wrapper torchmetrics-like metric to un-normalize input
    predictions and targets before metrics computation. This is
    typically useful for regression metrics when training is performed
    on normalized data.
    """
    def __init__(self, base_metric, mean, std):
        super().__init__()
        self.base_metric = base_metric  # The metric we want to apply
        self.mean = mean
        self.std = std
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # Unnormalize predictions and targets
        unnormalized_preds = preds * self.std + self.mean
        unnormalized_targets = targets * self.std + self.mean

        # Store them for reduction
        self.preds.append(unnormalized_preds)
        self.targets.append(unnormalized_targets)

    def compute(self):
        # Compute the base metric on the unnormalized values
        preds = torch.cat(self.preds, dim=0)
        targets = torch.cat(self.targets, dim=0)
        return self.base_metric(preds, targets)


class LogitMetric(torchmetrics.Metric):
    """Wrapper torchmetrics-like metric to un-normalize input
    predictions and targets before metrics computation. This is
    typically useful for regression metrics when training is performed
    on normalized data.
    """
    def __init__(self, base_metric):
        super().__init__()
        self.base_metric = base_metric  # The metric we want to apply
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = torch.nn.functional.sigmoid(preds)
        preds = preds / preds.sum(dim=1).unsqueeze(-1)

        print("Pred:", preds[0])
        print("Target:", targets[0])
        # Store them for reduction
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        # Compute the base metric on the unnormalized values
        preds = torch.cat(self.preds, dim=0)
        targets = torch.cat(self.targets, dim=0)
        return self.base_metric(preds, targets)


class PerDimMetric(torchmetrics.Metric):
    """Wrapper torchmetrics-like metric to un-normalize input
    predictions and targets before metrics computation. This is
    typically useful for regression metrics when training is performed
    on normalized data.
    """
    def __init__(self, metric_cls, dim_names: list, own_name="", *metric_args, **metric_kwargs):
        super().__init__()
        self.dim_names = dim_names  # List of names for each dimension
        self.own_name = own_name
        self.metrics = torch.nn.ModuleList(
            [metric_cls(*metric_args, **metric_kwargs) for _ in dim_names])

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        for i, name in enumerate(self.dim_names):
            self.metrics[i].update(preds[:, i], targets[:, i])  # Update per-dimension metric

    def compute(self):
        return {f"{name} {self.own_name}": self.metrics[i].compute() for i, name in enumerate(self.dim_names)}


class JSDivergence(torchmetrics.Metric):
    # class JSDivergence():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("pointwise_divs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        # self.reset()
        self.add_state("num_batches", default=torch.tensor(0.0), dist_reduce_fx="sum")

    """def reset(self):
        self.pointwise_divs = 0
        self.num_divs = 0"""

    @staticmethod
    def _pointwise_kl_divergence(p, q):
        # Following: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        return p * (p.log() - q.log())

    def update(self, logits, target):
        # Following https://pytorch.org/ignite/generated/ignite.metrics.JSDivergence.html
        # logits is an unnormalized logit
        # target is a distribution
        logits = logits.to("cpu")
        target = target.to("cpu")
        # Normalizing preds to distribution
        # This is the only part of the calculation that is not point-wise
        preds = torch.nn.functional.sigmoid(logits)
        preds = preds / preds.sum(dim=1).unsqueeze(-1)

        # In the JSDivergence, p: groundtruth/target, q: prediction/preds
        mi = (target + preds) / 2
        pm_div = self._pointwise_kl_divergence(target, mi)
        qm_div = self._pointwise_kl_divergence(preds, mi)

        self.pointwise_divs += (pm_div / 2 + qm_div / 2).sum()
        self.num_batches += len(pm_div)

    def compute(self):
        return {"JSDivergence": self.pointwise_divs / self.num_batches}
