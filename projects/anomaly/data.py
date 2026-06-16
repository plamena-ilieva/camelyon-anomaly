"""Зареждане и подготовка на данните от CAMELYON17.

CAMELYON17 са whole-slide изображения (`.tif` пирамиди) + XML анотации на
лезиите и CSV със slide-/patient-level етикети. За трениране извличаме патчове
(tiles) от слайдовете и ги етикетираме tumor/normal спрямо анотациите.

Тежките зависимости (`openslide`) се импортират лениво, за да могат тестовете и
лекият код да вървят без тях.
"""

import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Стандартна нормализация по ImageNet (полезна при transfer learning с VGG).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NORMAL, TUMOR = 0, 1


def default_transform(train: bool = False) -> transforms.Compose:
    """Трансформации за патч изображения.

    При ``train=True`` добавя леки аугментации (флипове, ротации), подходящи за
    хистопатология, при която ориентацията няма значение.
    """
    steps: list = []
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(steps)


def tissue_mask(rgb: np.ndarray, saturation_threshold: float = 0.05) -> np.ndarray:
    """Груба тъканна маска чрез наситеността в HSV.

    Фонът на хистологичните слайдове е почти бял (ниска наситеност). Връща
    булев масив, True там, където има тъкан. Праг по подразбиране е достатъчен;
    при нужда се заменя с Otsu (skimage.filters.threshold_otsu).
    """
    rgb = rgb.astype(np.float32) / 255.0
    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    saturation = np.where(maximum > 0, (maximum - minimum) / (maximum + 1e-8), 0.0)
    return saturation > saturation_threshold


def read_annotations(xml_path: str) -> list[np.ndarray]:
    """Чете полигоните на лезиите от CAMELYON XML анотация.

    Връща списък от масиви с форма (K, 2) -- (x, y) координати на полигоните.
    """
    tree = ET.parse(xml_path)
    polygons: list[np.ndarray] = []
    for annotation in tree.iter('Annotation'):
        coords = [(float(c.get('X')), float(c.get('Y'))) for c in annotation.iter('Coordinate')]
        if coords:
            polygons.append(np.array(coords, dtype=np.float32))
    return polygons


def point_in_polygons(x: float, y: float, polygons: list[np.ndarray]) -> bool:
    """Дали точка (x, y) е вътре в някой от полигоните (ray casting)."""
    for poly in polygons:
        inside = False
        n = len(poly)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and \
               (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        if inside:
            return True
    return False


def extract_tiles(slide_path: str,
                  tile_size: int = 96,
                  level: int = 0,
                  annotation_path: str | None = None,
                  max_tiles: int | None = None,
                  tissue_fraction: float = 0.5) -> list[tuple[np.ndarray, int]]:
    """Извлича патчове от един whole-slide image.

    Изисква ``openslide`` (импортира се лениво). За всеки патч с достатъчно
    тъкан връща (RGB масив, етикет), където етикетът е TUMOR, ако центърът на
    патча попада в анотиран полигон, иначе NORMAL.
    """
    import openslide  # ленив импорт -- тежка нативна зависимост

    slide = openslide.OpenSlide(slide_path)
    polygons = read_annotations(annotation_path) if annotation_path else []
    downsample = slide.level_downsamples[level]
    width, height = slide.level_dimensions[level]

    tiles: list[tuple[np.ndarray, int]] = []
    for top in range(0, height - tile_size, tile_size):
        for left in range(0, width - tile_size, tile_size):
            location = (int(left * downsample), int(top * downsample))
            region = slide.read_region(location, level, (tile_size, tile_size))
            rgb = np.array(region.convert('RGB'))

            if tissue_mask(rgb).mean() < tissue_fraction:
                continue

            center_x = (left + tile_size / 2) * downsample
            center_y = (top + tile_size / 2) * downsample
            label = TUMOR if point_in_polygons(center_x, center_y, polygons) else NORMAL
            tiles.append((rgb, label))

            if max_tiles is not None and len(tiles) >= max_tiles:
                slide.close()
                return tiles
    slide.close()
    return tiles


class PatchDataset(Dataset):
    """Датасет от RGB патчове и бинарни етикети (NORMAL/TUMOR).

    Приема списък от numpy масиви (или 4D масив N x H x W x 3) и съответните
    етикети. Не зависи от openslide, така че тестовете вървят със синтетични
    данни.
    """

    def __init__(self,
                 patches: list[np.ndarray] | np.ndarray,
                 labels: list[int] | np.ndarray,
                 transform: transforms.Compose | None = None) -> None:
        self.patches = list(patches)
        self.labels = list(labels)
        if len(self.patches) != len(self.labels):
            raise ValueError('Броят патчове и етикети трябва да съвпада.')
        self.transform = transform or default_transform(train=False)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        patch = self.patches[index]
        image = self.transform(patch.astype(np.uint8))
        return image, int(self.labels[index])

    @classmethod
    def from_directory(cls,
                       root: str,
                       transform: transforms.Compose | None = None) -> 'PatchDataset':
        """Зарежда кеширани патчове от ``root/normal`` и ``root/tumor``."""
        from PIL import Image

        patches: list[np.ndarray] = []
        labels: list[int] = []
        for name, label in (('normal', NORMAL), ('tumor', TUMOR)):
            for path in sorted(glob.glob(os.path.join(root, name, '*.png'))):
                patches.append(np.array(Image.open(path).convert('RGB')))
                labels.append(label)
        return cls(patches, labels, transform)
