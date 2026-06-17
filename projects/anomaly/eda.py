"""Разглеждане на данните (стъпка 2) -- статистики и визуализации.

Функциите връщат стойности (за тестване и за вграждане в доклада) и по желание
рисуват фигура чрез matplotlib. Коментарите към резултатите се пишат в
``docs/MODEL_REPORT.md``.
"""

import numpy as np


def class_distribution(labels: list[int] | np.ndarray) -> dict[int, int]:
    """Брой наблюдения по клас (проверка за дисбаланс)."""
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def pixel_statistics(patches: list[np.ndarray] | np.ndarray) -> dict[str, list[float]]:
    """Средно и стандартно отклонение на пикселите по канал (R, G, B).

    Полезно за избор на нормализация и за откриване на цветови вариации между
    слайдове (различно оцветяване -- честа аномалия в хистопатологията).

    Усреднява се в ``float64``: при милиони пиксели float32 акумулаторът губи
    точност и каналните средни излизат грешни (дори идентични).
    """
    stacked = np.stack([np.asarray(p, dtype=np.float32) / 255.0 for p in patches])
    mean = stacked.mean(axis=(0, 1, 2), dtype=np.float64)
    std = stacked.std(axis=(0, 1, 2), dtype=np.float64)
    return {'mean': mean.tolist(), 'std': std.tolist()}


def summarize(patches: list[np.ndarray] | np.ndarray, labels: list[int] | np.ndarray) -> dict:
    """Обобщен отчет: брой наблюдения, форма, разпределение, пикселна статистика."""
    patches = list(patches)
    sample = np.asarray(patches[0])
    return {
        'num_observations': len(patches),
        'patch_shape': tuple(sample.shape),
        'dtype': str(sample.dtype),
        'class_distribution': class_distribution(labels),
        'pixel_statistics': pixel_statistics(patches),
    }


def plot_class_distribution(labels: list[int] | np.ndarray, filename: str) -> None:
    """Стълбовидна диаграма на разпределението по класове -> файл."""
    import matplotlib.pyplot as plt

    dist = class_distribution(labels)
    names = {0: 'normal', 1: 'tumor'}
    fig, ax = plt.subplots()
    ax.bar([names.get(k, str(k)) for k in dist], list(dist.values()))
    ax.set_ylabel('брой патчове')
    ax.set_title('Разпределение по класове')
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def plot_sample_grid(patches: list[np.ndarray],
                     labels: list[int],
                     filename: str,
                     nrows: int = 3,
                     ncols: int = 6) -> None:
    """Решетка с примерни патчове и техните етикети -> файл."""
    import matplotlib.pyplot as plt

    names = {0: 'normal', 1: 'tumor'}
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    for ax, patch, label in zip(axs.flat, patches, labels):
        ax.imshow(np.asarray(patch).astype(np.uint8))
        ax.set_title(names.get(label, str(label)), fontsize=8)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
