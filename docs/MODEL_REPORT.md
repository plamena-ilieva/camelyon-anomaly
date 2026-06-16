# Model Report — Разпознаване на медицинска аномалия (CAMELYON17)

> Задължителен deliverable. Попълвай разделите с резултати, фигури и коментари,
> докато напредваш по стъпките. Запазвай фигурите в `docs/figures/`.

## 1. Задача и постановка

- **Цел:** засичане на метастази (tumor) в хистопатологични патчове от лимфни
  възли (CAMELYON17) — бинарна класификация *normal* / *tumor*, както и
  unsupervised засичане на аномалии чрез реконструкция.
- **Защо е аномалия:** tumor тъканта е рядка и морфологично различна от
  нормалната → проблем на anomaly / out-of-distribution detection.

## 2. Научни статии и техники (стъпка 1)

| Източник | Принос за проекта |
|----------|-------------------|
| Litjens et al., *A survey on deep learning in medical image analysis* | общ преглед |
| CAMELYON16/17 challenge papers (Bejnordi et al., Bandi et al.) | постановка, метрики, baseline |
| Veeling et al., *Rotation Equivariant CNNs* (PatchCamelyon) | патч-базиран подход |
| Ronneberger et al., *U-Net* | автоенкодер/сегментация |
| Simonyan & Zisserman, *Very Deep Conv. Networks (VGG)* | архитектура VGG |
| Anomaly detection survey (Wikipedia + обзори) | категоризация на методите |

**Категории методи за anomaly detection:** supervised (класификатор), 
semi-/unsupervised (autoencoder reconstruction error, one-class), 
reconstruction-based, distance-/density-based. _(разпиши с 2-3 изречения всяка)_

### Метрика Cohen's Kappa

κ = (p_o − p_e) / (1 − p_e), където p_o = наблюдавана съгласуваност (accuracy),
p_e = очаквана по случайност. Защо тук: датасетът е дисбалансиран → точността
подвежда; Kappa коригира спрямо случайността. Тълкуване (Landis & Koch):
< 0 лошо, 0–0.2 слабо, 0.2–0.4 приемливо, 0.4–0.6 умерено, 0.6–0.8 значително,
0.8–1.0 почти пълно съгласие. Имплементация: `projects/anomaly/metrics.py`.

## 3. Разглеждане на данните (стъпка 2)

- Брой наблюдения (патчове) общо и по сплит: _попълни_
- Форма и тип на характеристиките (RGB патчове H×W×3, uint8): _попълни_
- Разпределение по класове (дисбаланс): _фигура `figures/class_distribution.png`_
- Пикселна статистика по канал (вариации в оцветяването между слайдове): _попълни_
- Примерни патчове: _фигура `figures/sample_grid.png`_
- Откриване на аномалии/артефакти в данните (размазване, мастило, фон): _коментар_

_Използвай `projects/anomaly/eda.py` (summarize, plot_class_distribution, plot_sample_grid)._

## 4. CNN експерименти (стъпка 3)

- Архитектура: `models.SimpleCNN`
- Хиперпараметри, крива на загубата, accuracy / Cohen Kappa: _таблица + фигура_

## 5. VGG варианти (стъпка 4)

- VGG11 / VGG13 / VGG16 (`models.VGG`), със/без BatchNorm, от нула vs предобучен
- Сравнителна таблица по Cohen Kappa: _попълни_

## 6. Автоенкодер U-Net (стъпка 5)

- `models.UNetAutoencoder`, трениран само на normal патчове
- Anomaly score = MSE на реконструкцията (`train.reconstruction_error`)
- Праг и ROC/AUC за разделяне normal/tumor: _фигура_

## 7. Сравнение и изводи

| Модел | Accuracy | Cohen Kappa | Бележки |
|-------|----------|-------------|---------|
| SimpleCNN | | | |
| VGG11 | | | |
| VGG16 | | | |
| U-Net (recon.) | | | |

## 8. Потребителски интерфейс (стъпка 6)

- Streamlit: `projects/anomaly/app.py` — качване на патч → предсказание + вероятност.
- Тестове: `tests/test_anomaly_*.py` (BDD стил, Arrange/Act/Assert).

## 9. Възпроизводимост

- Среда: Python 3.12, зависимости в `requirements.txt`.
- Трениране: Google Colab (GPU). Сийдове, версии, команди: _попълни_.
