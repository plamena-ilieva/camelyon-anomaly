# Разпознаване на медицинска аномалия (CAMELYON17)

Курсов проект по Deep Learning: уеб апликация за засичане на метастази в лимфни
възли върху хистопатологични whole-slide изображения от датасета **CAMELYON17**.

Поддържа три подхода: CNN, VGG (11/16) и U-Net автоенкодер за unsupervised
засичане на аномалии. Основна метрика: **Cohen's Kappa**.

## Структура

```
projects/anomaly/      # пакетът на проекта
  metrics.py           # Cohen Kappa, confusion matrix, accuracy
  data.py              # WSI tiling (openslide), tissue mask, анотации, PatchDataset
  eda.py               # разглеждане на данните + визуализации
  models.py            # SimpleCNN, VGG, UNetAutoencoder
  train.py             # тренировъчен цикъл, оценка, anomaly score
  app.py               # Streamlit UI
  colab_camelyon.ipynb # end-to-end pipeline за Colab (GPU)
  PLAN.md              # пълен план
tests/                 # BDD тестове (Arrange/Act/Assert)
docs/MODEL_REPORT.md   # доклад за модела
```

## Локално стартиране

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m pytest tests/ -q              # тестове
streamlit run projects/anomaly/app.py   # уеб приложение
```

## Колаб

Отвори `projects/anomaly/colab_camelyon.ipynb` в Google Colab (GPU runtime) и в
клетка 2 сложи URL-а на това repo. При промяна: `git push` локално → повтори
клетка 2 в Colab (прави `git pull`).
