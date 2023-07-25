import torch
from torch import Tensor
import matplotlib.pyplot as plt
from typing import Any, Optional


def rescale_kpt(kpt: Tensor, size):
    img_w, img_h = size

    rescaled_kpt = kpt * torch.tensor([img_w, img_h, 1], dtype=torch.float32)

    return rescaled_kpt


def get_kpt_mark(rescaled_kpt: Tensor, color):
    x, y, _ = rescaled_kpt

    circle = plt.Circle((x, y), 3, color=color)

    return circle


def get_kpt_text(rescaled_kpt: Tensor, kpt_class: str, visibility: Optional[Any] = None):
    x, y, _ = rescaled_kpt

    txt = kpt_class if visibility is None else f'{kpt_class}: {visibility:0.2f}'

    text = plt.text(x, y, txt, fontsize=7, bbox=dict(
        facecolor='yellow', alpha=0.5))

    return text
