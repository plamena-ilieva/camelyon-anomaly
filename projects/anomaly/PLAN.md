# Проект 2 — Разпознаване на медицинска аномалия (CAMELYON17)

Уеб апликация за засичане на метастази в лимфни възли върху хистопатологични
whole-slide изображения от датасета **CAMELYON17**.

## Структура на файловете

```
projects/anomaly/
├── PLAN.md            # този файл
├── __init__.py        # реекспорт на публичното API
├── data.py            # зареждане на WSI, tiling, tissue mask, анотации, PatchDataset
├── eda.py             # статистики и визуализации за стъпка 2
├── models.py          # SimpleCNN, VGG (конфигурируем), UNetAutoencoder
├── metrics.py         # Cohen Kappa (от нула) + accuracy
├── train.py           # тренировъчен цикъл + оценка
└── app.py             # Streamlit потребителски интерфейс

tests/
├── test_anomaly_metrics.py   # Cohen Kappa, accuracy
├── test_anomaly_models.py    # форми на изходите на моделите
├── test_anomaly_data.py      # PatchDataset, tiling помощни
└── test_anomaly_app.py       # логика на UI (предсказание върху патч)

docs/
└── MODEL_REPORT.md    # задължителен deliverable
```

## Стъпки от заданието → къде се реализират

| # | Стъпка | Файл / артефакт |
|---|--------|-----------------|
| 1 | Статии + Cohen Kappa | `docs/MODEL_REPORT.md`, `metrics.py` |
| 2 | Разглеждане на данните (EDA) | `eda.py`, EDA notebook |
| 3 | CNN експерименти | `models.SimpleCNN`, `train.py` |
| 4 | VGG варианти | `models.VGG` (VGG11/13/16 конфигурации) |
| 5 | Автоенкодер (U-Net) | `models.UNetAutoencoder` |
| 6 | Streamlit UI + тестове | `app.py`, `tests/` |
| 7 | Презентация | `docs/` (отделен файл) |

## Реалност за данните (важно)

Пълният CAMELYON17 е ~2.7 TB whole-slide `.tif` пирамиди + XML анотации на лезии
и CSV със slide-/patient-level етикети (pN-stage). Сваляне на всичко в Colab е
непрактично. Затова пайплайнът работи с **подмножество от слайдове**:

1. Сваляме няколко слайда (вкл. tumor слайдове с анотации) от grand-challenge.
2. С `openslide` + Otsu tissue mask извличаме патчове на ниво с разумна резолюция.
3. Етикетираме патч като *tumor*, ако центърът му попада в анотиран полигон, иначе
   *normal*. Това възпроизвежда задачата на ниво патч (както PCam).
4. Балансираме и кешираме патчовете на диск → бързо трениране в Colab.

Тестовете НЕ зависят от свалянето: ползват малки синтетични тензори/изображения,
така че `pytest` върви навсякъде (BDD изискване на курса).

## Anomaly detection — два подхода

- **Supervised (CNN / VGG):** бинарна класификация патч tumor/normal. Основна метрика
  Cohen Kappa (съгласуваност отвъд случайността), плюс accuracy, AUC.
- **Unsupervised (U-Net автоенкодер):** трениране за реконструкция само на *normal*
  патчове; аномалията се засича по висока грешка на реконструкцията (anomaly score).
  U-Net може да се ползва и като сегментационна мрежа за tumor маска.

## Ред на изпълнение

1. ✅ Скелет + план (този етап)
2. metrics.py + тестове (бързо, без данни)
3. models.py + тестове (бързо, без данни)
4. data.py + eda.py (нужни данни/Colab)
5. train.py + експерименти в Colab notebook
6. app.py + тестове
7. MODEL_REPORT.md + презентация
