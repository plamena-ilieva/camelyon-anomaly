"""Метрики за оценка на класификацията.

Основната метрика по заданието е **Cohen's Kappa** -- измерва съгласуваността
между предсказани и истински етикети, коригирана спрямо съгласуваност по
случайност. Имплементирана е от нула, за да може да се обясни на защитата;
`torchmetrics.CohenKappa` / `sklearn.metrics.cohen_kappa_score` дават същото.
"""

import torch


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Връща матрица на обърквания с форма (num_classes, num_classes).

    Ред = истински клас, стълб = предсказан клас.
    """
    y_true = y_true.flatten().long()
    y_pred = y_pred.flatten().long()
    indices = y_true * num_classes + y_pred
    matrix = torch.bincount(indices, minlength=num_classes**2)
    return matrix.reshape(num_classes, num_classes)


def cohen_kappa(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int = 2) -> float:
    r"""Cohen's Kappa коефициент.

    .. math:: \kappa = \frac{p_o - p_e}{1 - p_e}

    където ``p_o`` е наблюдаваната съгласуваност (accuracy), а ``p_e`` е
    очакваната съгласуваност по случайност. Стойности: 1 = пълна съгласуваност,
    0 = на ниво случайност, < 0 = по-лошо от случайност.
    """
    cm = confusion_matrix(y_true, y_pred, num_classes).float()
    total = cm.sum()
    if total == 0:
        return 0.0

    observed_agreement = torch.diagonal(cm).sum() / total

    row_marginals = cm.sum(dim=1) / total
    col_marginals = cm.sum(dim=0) / total
    expected_agreement = (row_marginals * col_marginals).sum()

    denominator = 1.0 - expected_agreement
    if denominator == 0:
        return 1.0
    return ((observed_agreement - expected_agreement) / denominator).item()


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Дял на верните предсказания."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    if y_true.numel() == 0:
        return 0.0
    return (y_true == y_pred).float().mean().item()
