"""Тренировъчен цикъл и оценка за класификаторите (CNN / VGG).

Връща история на загубите и метриките по епохи, така че да се визуализират в
доклада и да се сравняват архитектурите.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader

from projects.anomaly import metrics


def get_device() -> torch.device:
    """Връща CUDA, ако е налична (Colab GPU), иначе CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device | None = None) -> dict[str, float]:
    """Оценява модела върху loader -> {'accuracy', 'cohen_kappa'}."""
    device = device or get_device()
    model.eval()
    all_true: list[torch.Tensor] = []
    all_pred: list[torch.Tensor] = []
    for images, labels in loader:
        logits = model(images.to(device))
        all_pred.append(logits.argmax(dim=1).cpu())
        all_true.append(labels)

    y_true = torch.cat(all_true)
    y_pred = torch.cat(all_pred)
    return {
        'accuracy': metrics.accuracy(y_true, y_pred),
        'cohen_kappa': metrics.cohen_kappa(y_true, y_pred, num_classes=2),
    }


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, device: torch.device) -> float:
    """Една епоха обучение; връща средната загуба."""
    model.train()
    running = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / max(len(loader), 1)


def fit(model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        num_epochs: int = 10,
        lr: float = 1e-3,
        device: torch.device | None = None) -> dict[str, list]:
    """Обучава класификатор и връща история по епохи.

    История: {'train_loss': [...], 'val_accuracy': [...], 'val_cohen_kappa': [...]}.
    """
    device = device or get_device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: dict[str, list] = {'train_loss': [], 'val_accuracy': [], 'val_cohen_kappa': []}
    for _ in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)
        if val_loader is not None:
            scores = evaluate(model, val_loader, device)
            history['val_accuracy'].append(scores['accuracy'])
            history['val_cohen_kappa'].append(scores['cohen_kappa'])
    return history


@torch.no_grad()
def reconstruction_error(model: nn.Module,
                         images: torch.Tensor,
                         device: torch.device | None = None) -> torch.Tensor:
    """Anomaly score за U-Net автоенкодера: MSE на реконструкцията на изображение.

    Висока стойност => вероятна аномалия (моделът е виждал само normal патчове).
    """
    device = device or get_device()
    model.eval()
    images = images.to(device)
    reconstructed = model(images)
    return ((reconstructed - images)**2).mean(dim=(1, 2, 3)).cpu()
