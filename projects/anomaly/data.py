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

    При ``train=True`` добавя аугментации:

    * флипове и ротации -- ориентацията на патча няма значение;
    * ``ColorJitter`` (stain augmentation) -- H&E оцветяването варира между
      лаборатории/пациенти; това учи модела да е инвариантен към цвета вместо
      да "познава слайда", което е честа причина за слаба генерализация.

    Входът е numpy ``uint8`` масив (H x W x 3). ``ToTensor`` е първо, защото
    новите версии на torchvision искат аугментациите да работят върху тензор
    (или PIL), не върху numpy масив.
    """
    steps: list = [transforms.ToTensor()]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ]
    steps += [transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
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


def extract_tiles_balanced(slide_path: str,
                           tile_size: int = 96,
                           level: int = 0,
                           annotation_path: str | None = None,
                           n_tumor: int = 1000,
                           n_normal: int = 1000,
                           tissue_fraction: float = 0.5,
                           max_attempts_factor: int = 50,
                           seed: int = 0,
                           verbose: bool = False,
                           log_every: int = 200) -> list[tuple[np.ndarray, int]]:
    """Насочено семплиране на патчове от един whole-slide image.

    За разлика от :func:`extract_tiles` (последователно сканиране с лимит, което
    пропуска малките лезии), тук:

    * TUMOR патчовете се семплират с център **вътре в анотираните полигони** ->
      гарантирано попадаме в лезията.
    * NORMAL патчовете се семплират на случайни тъканни места **извън** всички
      полигони.

    Подава се случайно семплиране (не пълно сканиране), затова е бързо дори за
    слайдове от по няколко GB. Координатите на openslide ``read_region`` са в
    референтна система на ниво 0, затова работим в ниво-0 пиксели.
    """
    import openslide  # ленив импорт -- тежка нативна зависимост

    rng = np.random.default_rng(seed)
    slide = openslide.OpenSlide(slide_path)
    downsample = slide.level_downsamples[level]
    width0, height0 = slide.level_dimensions[0]
    span = int(tile_size * downsample)  # ниво-0 пиксели, покрити от един патч
    polygons = read_annotations(annotation_path) if annotation_path else []
    name = os.path.basename(slide_path)

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    log(f'[{name}] ниво={level} (downsample x{downsample:.0f}), ниво-0 {width0}x{height0}px, '
        f'патч={span}px@ниво-0, полигони={len(polygons)}')

    def read_tile(x0: float, y0: float) -> np.ndarray:
        region = slide.read_region((int(x0), int(y0)), level, (tile_size, tile_size))
        return np.array(region.convert('RGB'))

    tiles: list[tuple[np.ndarray, int]] = []

    # --- TUMOR: центрове вътре в полигоните ---
    if polygons and n_tumor > 0:
        kept = attempts = 0
        max_attempts = n_tumor * max_attempts_factor
        while kept < n_tumor and attempts < max_attempts:
            attempts += 1
            poly = polygons[rng.integers(len(polygons))]
            cx = rng.uniform(poly[:, 0].min(), poly[:, 0].max())
            cy = rng.uniform(poly[:, 1].min(), poly[:, 1].max())
            if not point_in_polygons(cx, cy, [poly]):
                continue
            x0, y0 = cx - span / 2, cy - span / 2
            if x0 < 0 or y0 < 0 or x0 + span >= width0 or y0 + span >= height0:
                continue
            rgb = read_tile(x0, y0)
            if tissue_mask(rgb).mean() >= tissue_fraction:
                tiles.append((rgb, TUMOR))
                kept += 1
                if kept % log_every == 0:
                    log(f'[{name}]   tumor {kept}/{n_tumor} (опити: {attempts})')
        log(f'[{name}] tumor: {kept} патча от {attempts} опита')
        if kept < n_tumor:
            log(f'[{name}] ВНИМАНИЕ: само {kept}/{n_tumor} tumor патча (лезията е малка '
                f'или max_attempts_factor е нисък)')
    elif n_tumor > 0 and not polygons:
        log(f'[{name}] няма полигони -> 0 tumor патча (очаквано за normal слайд)')

    # --- NORMAL: случайни тъканни места извън полигоните ---
    if n_normal > 0:
        kept = attempts = 0
        max_attempts = n_normal * max_attempts_factor
        while kept < n_normal and attempts < max_attempts:
            attempts += 1
            x0 = rng.integers(0, max(1, width0 - span))
            y0 = rng.integers(0, max(1, height0 - span))
            cx, cy = x0 + span / 2, y0 + span / 2
            if polygons and point_in_polygons(cx, cy, polygons):
                continue
            rgb = read_tile(x0, y0)
            if tissue_mask(rgb).mean() >= tissue_fraction:
                tiles.append((rgb, NORMAL))
                kept += 1
                if kept % log_every == 0:
                    log(f'[{name}]   normal {kept}/{n_normal} (опити: {attempts})')
        log(f'[{name}] normal: {kept} патча от {attempts} опита')

    slide.close()
    return tiles


def grouped_train_val_split(labels: list[int] | np.ndarray,
                            groups: list | np.ndarray,
                            val_fraction: float = 0.3,
                            seed: int = 0) -> tuple[list[int], list[int]]:
    """Train/val индекси **без изтичане**: нито една група (слайд/пациент) не е
    едновременно в train и val.

    Това е истинският тест за генерализация при CAMELYON -- ако патчове от един
    и същ слайд попаднат и в train, и в val, моделът може да "познава слайда" по
    оцветяването вместо да засича тумора, и метриките излизат фалшиво високи.

    Разделя поотделно за всеки клас, за да присъстват и двата класа във val.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in np.unique(labels):
        cls_groups = np.unique(groups[labels == cls])
        rng.shuffle(cls_groups)
        n_val_groups = max(1, round(len(cls_groups) * val_fraction))
        val_groups = set(cls_groups[:n_val_groups].tolist())
        for i in np.where(labels == cls)[0]:
            (val_idx if groups[i] in val_groups else train_idx).append(int(i))
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


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
