import torch
from torch import Tensor
import matplotlib.pyplot as plt
import random
from typing import Optional

from .misc import nested_tensor_from_tensor_list
from .kpts import get_kpt_text, rescale_kpt


def show_pred(
        x: tuple[Tensor, Tensor] | Tensor,
        y: dict[str, Tensor],
        keypoints: list[str],
        index: Optional[int] = None
):
    if not isinstance(x, tuple):
        x = nested_tensor_from_tensor_list(x)

    images, masks = x

    if index is None:
        batch_index = random.randint(0, images.size(0)-1)
    else:
        batch_index = index

    image = images[batch_index]
    mask = masks[batch_index]

    pred_kpts = y['pred_kpts'][batch_index]
    pred_visibility = y['pred_visibility'][batch_index]
    kpts = torch.cat([pred_kpts, pred_visibility], dim=-1)

    not_mask = ~mask
    H = not_mask[0].nonzero()[-1].item() + 1
    W = not_mask[:, 0].nonzero()[-1].item() + 1
    size = (H, W)

    rescaled_kpts = rescale_kpt(kpts, size)

    _, ax = plt.subplots()
    ax.imshow(image.permute(1, 2, 0))

    for kpt, kpt_class in zip(rescaled_kpts, keypoints):
        _, _, visibility = kpt

        kpt_text = get_kpt_text(kpt, kpt_class, visibility)
        ax.add_artist(kpt_text)
