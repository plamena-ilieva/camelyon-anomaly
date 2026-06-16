# Проект 2 — Разпознаване на медицинска аномалия (CAMELYON17)

Уеб апликация за засичане на метастази в лимфни възли върху хистопатологични
whole-slide изображения от датасета **CAMELYON17**.

Виж [PLAN.md](PLAN.md) за пълния план и [docs/MODEL_REPORT.md](../../docs/MODEL_REPORT.md)
за доклада.

## Структура

| Файл | Съдържание |
|------|-----------|
| `metrics.py` | Cohen Kappa (от нула), confusion matrix, accuracy |
| `data.py` | WSI tiling (openslide), tissue mask, XML анотации, `PatchDataset` |
| `eda.py` | разглеждане на данните + визуализации |
| `models.py` | `SimpleCNN`, `VGG` (11/13/16), `UNetAutoencoder` |
| `train.py` | тренировъчен цикъл, оценка по Cohen Kappa, anomaly score |
| `app.py` | Streamlit потребителски интерфейс |
| `colab_camelyon.ipynb` | end-to-end pipeline за Google Colab (GPU) |

## Локално стартиране

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# тестове
python -m pytest tests/test_anomaly_*.py -q

# Streamlit приложение
streamlit run projects/anomaly/app.py
```

## Работен поток с Colab

Кодът се редактира локално и се синхронизира с Colab през git:

```bash
# еднократно (локално)
git init
git add .
git commit -m "Anomaly detection project"
git remote add origin https://github.com/<потребител>/<repo>.git
git push -u origin main
```

```python
# в началото на всяка Colab сесия — тегли най-новата версия
!git -C /content/DL_25-26 pull
```

При промяна по кода: `git push` локално → `!git pull` в Colab.
