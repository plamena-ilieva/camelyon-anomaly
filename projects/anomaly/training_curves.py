"""Бързи криви train vs val (загуба и Cohen Kappa) за доклада/презентацията.

Ползва кешираните патчове от голямия run (`camelyon_run/chunks/`), взема подизвадка
за скорост и обучава кратко `SimpleCNN`, като записва метриките по епохи. Резултатът
са две илюстративни диаграми, които покриват изискването за `train vs val` графики.

Стартиране в Colab:  ``%run projects/anomaly/training_curves.py``
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from projects.anomaly import data, metrics, models, train


def load_chunks(chunks_dir: str) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Зарежда кешираните патчове (patches, labels, groups) от папка с .npz чанкове."""
    patches: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    for name in sorted(os.listdir(chunks_dir)):
        if not name.endswith('.npz'):
            continue
        d = np.load(os.path.join(chunks_dir, name), allow_pickle=True)
        patches.extend(list(d['patches']))
        labels.extend(list(d['labels']))
        groups.extend([str(d['patient'])] * len(d['labels']))
    return patches, np.array(labels), np.array(groups)


@torch.no_grad()
def _evaluate_loss_and_kappa(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                             device: torch.device) -> tuple[float, float]:
    """Връща (средна загуба, Cohen Kappa) върху loader."""
    model.eval()
    total = 0.0
    trues, preds = [], []
    for images, targets in loader:
        out = model(images.to(device))
        total += criterion(out, targets.to(device)).item()
        preds.append(out.argmax(1).cpu())
        trues.append(targets)
    kappa = metrics.cohen_kappa(torch.cat(trues), torch.cat(preds), num_classes=2)
    return total / len(loader), kappa


def train_with_history(patches,
                       labels,
                       groups,
                       *,
                       num_epochs: int = 12,
                       lr: float = 5e-4,
                       subsample: int = 8000,
                       seed: int = 23) -> dict[str, list]:
    """Обучава SimpleCNN върху подизвадка, записвайки загуба и Cohen Kappa по епохи."""
    rng = np.random.default_rng(seed)
    if subsample and subsample < len(patches):
        sel = rng.choice(len(patches), size=subsample, replace=False)
        patches = [patches[i] for i in sel]
        labels, groups = labels[sel], groups[sel]

    train_idx, val_idx = data.grouped_train_val_split(labels, groups, val_fraction=0.25, seed=seed)
    train_loader = DataLoader(Subset(
        data.PatchDataset(patches, labels, transform=data.default_transform(train=True)),
        train_idx),
                              batch_size=128,
                              shuffle=True,
                              num_workers=2)
    val_loader = DataLoader(Subset(
        data.PatchDataset(patches, labels, transform=data.default_transform(train=False)),
        val_idx),
                            batch_size=128,
                            shuffle=False,
                            num_workers=2)

    device = train.get_device()
    model = models.SimpleCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: dict[str, list] = {
        'train_loss': [],
        'val_loss': [],
        'train_kappa': [],
        'val_kappa': []
    }
    for epoch in range(num_epochs):
        model.train()
        for images, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
        train_loss, train_kappa = _evaluate_loss_and_kappa(model, train_loader, criterion, device)
        val_loss, val_kappa = _evaluate_loss_and_kappa(model, val_loader, criterion, device)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_kappa'].append(train_kappa)
        history['val_kappa'].append(val_kappa)
        print(
            f'epoch {epoch}: train_loss {train_loss:.3f} | val_loss {val_loss:.3f} | '
            f'train κ {train_kappa:.3f} | val κ {val_kappa:.3f}',
            flush=True)
    return history


def plot_curves(history: dict[str, list], out_dir: str) -> None:
    """Записва две диаграми: train vs val загуба и train vs val Cohen Kappa."""
    epochs = list(range(len(history['train_loss'])))
    for key_train, key_val, ylabel, title, fname in [
        ('train_loss', 'val_loss', 'загуба', 'Train vs val загуба (SimpleCNN)', 'curve_loss.png'),
        ('train_kappa', 'val_kappa', 'Cohen Kappa', 'Train vs val Cohen Kappa (SimpleCNN)',
         'curve_kappa.png'),
    ]:
        plt.figure(figsize=(5, 3.5))
        plt.plot(epochs, history[key_train], marker='o', label='train')
        plt.plot(epochs, history[key_val], marker='o', label='val')
        plt.xlabel('епоха')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close()
    print('Записани: curve_loss.png и curve_kappa.png в', out_dir)


def main(out_dir: str = '/content/drive/MyDrive/camelyon_run') -> None:
    chunks = os.path.join(out_dir, 'chunks')
    patches, labels, groups = load_chunks(chunks)
    print('Заредени патчове:', len(patches))
    history = train_with_history(patches, labels, groups)
    plot_curves(history, out_dir)


if __name__ == '__main__':  # pragma: no cover
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except ImportError:
        pass
    main()
