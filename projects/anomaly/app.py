"""Streamlit потребителски интерфейс (стъпка 6).

Качваш патч/изображение от хистологичен слайд и моделът предсказва дали в него
има метастаза (tumor) или е нормална тъкан, заедно с вероятност / anomaly score.

Стартиране:  ``streamlit run projects/anomaly/app.py``

Логиката на предсказването (``predict_patch``) е изнесена отделно, за да се
тества без да се вдига Streamlit (виж tests/test_anomaly_app.py).
"""

import numpy as np
import torch
from torch import nn

from projects.anomaly.data import default_transform

CLASS_NAMES = {0: 'normal (нормална тъкан)', 1: 'tumor (метастаза)'}


def predict_patch(model: nn.Module, patch: np.ndarray) -> dict[str, float]:
    """Предсказва клас и вероятности за един RGB патч.

    Връща {'label': int, 'prob_tumor': float, 'prob_normal': float}.
    """
    model.eval()
    transform = default_transform(train=False)
    tensor = transform(np.asarray(patch).astype(np.uint8)).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze(0)
    return {
        'label': int(probs.argmax().item()),
        'prob_normal': float(probs[0].item()),
        'prob_tumor': float(probs[1].item()),
    }


def main() -> None:  # pragma: no cover -- интерактивен Streamlit вход
    import streamlit as st
    from PIL import Image

    st.set_page_config(page_title='Разпознаване на медицинска аномалия', page_icon='🔬')
    st.title('🔬 Разпознаване на метастази (CAMELYON17)')
    st.write('Качи патч от хистологичен слайд. Моделът предсказва tumor / normal.')

    model_path = st.sidebar.text_input('Път до модел (.pt)', 'model.pt')
    arch_hint = st.sidebar.selectbox('Архитектура (ако не е записана в .pt)',
                                     ['VGG11', 'VGG13', 'VGG16', 'SimpleCNN'])

    uploaded = st.file_uploader('Изображение (PNG/JPG)', type=['png', 'jpg', 'jpeg'])
    if uploaded is None:
        st.info('Очаквам изображение...')
        return

    image = Image.open(uploaded).convert('RGB')
    st.image(image, caption='Вход', width=256)

    model, arch = _load_model(model_path, arch_hint)
    if model is None:
        st.warning('Няма зареден модел. Посочи валиден .pt файл в страничната лента.')
        return
    st.caption(f'Архитектура: {arch}')

    result = predict_patch(model, np.array(image))
    st.subheader(f'Резултат: **{CLASS_NAMES[result["label"]]}**')
    st.metric('Вероятност за tumor', f'{result["prob_tumor"]:.1%}')
    st.progress(result['prob_tumor'])


def _build_model(arch: str) -> nn.Module:
    """Създава модел по име на архитектура."""
    from projects.anomaly.models import VGG, SimpleCNN

    return SimpleCNN() if arch == 'SimpleCNN' else VGG(config=arch)


def _load_model(path: str, arch_hint: str = 'VGG11') -> tuple[nn.Module | None, str | None]:
    """Зарежда модел от ``.pt``.

    Поддържа два формата: суров ``state_dict`` (ползва ``arch_hint``) или
    чекпойнт-речник ``{'arch': ..., 'state_dict': ...}`` (записан от notebook-а).
    """
    import os

    if not os.path.exists(path):
        return None, None
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        arch = checkpoint.get('arch', arch_hint)
        state = checkpoint['state_dict']
    else:
        arch, state = arch_hint, checkpoint
    model = _build_model(arch)
    model.load_state_dict(state)
    return model, arch


if __name__ == '__main__':  # pragma: no cover
    main()
