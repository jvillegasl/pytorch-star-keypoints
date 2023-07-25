from torch import Tensor
import matplotlib.pyplot as plt
import random
from typing import Optional

from .bbox import rescale_bboxes, get_bbox_patch, get_bbox_text
from .kpts import get_kpt_mark, rescale_kpt, get_kpt_text


def show_sample(
        xb: tuple[Tensor, Tensor],
        yb: tuple,
        classes: list[str],
        keypoints: dict[str, list[str]],
        index: Optional[int] = None
):
    images, masks = xb

    if index is None:
        batch_index = random.randint(0, images.size(0)-1)
    else:
        batch_index = index

    image = images[batch_index]
    mask = masks[batch_index]
    data = yb[batch_index]

    labels = data['labels']

    not_mask = ~mask
    H = not_mask[0].nonzero()[-1].item() + 1
    W = not_mask[:, 0].nonzero()[-1].item() + 1
    size = (H, W)

    bboxes = data['boxes']
    kpts = data['keypoints']

    rescaled_bboxes = rescale_bboxes(bboxes, size)
    rescaled_kpts = rescale_kpt(kpts, size)

    colors = []
    for _ in classes:
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)
        colors.append(color)

    _, ax = plt.subplots()
    ax.imshow(image.permute(1, 2, 0))

    for bbox, label, kpts in zip(rescaled_bboxes, labels, rescaled_kpts):
        bbox_patch = get_bbox_patch(bbox, colors[label])
        ax.add_patch(bbox_patch)

        bbox_text = get_bbox_text(bbox, classes[label])
        ax.add_artist(bbox_text)

        kpts_classes = keypoints[classes[label]]
        for kpt, kpt_class in zip(kpts, kpts_classes):
            _, _, visibility = kpt

            if visibility != 1:
                pass

            kpt_mark = get_kpt_mark(kpt, colors[label])
            ax.add_patch(kpt_mark)

            kpt_text = get_kpt_text(kpt, kpt_class)
            ax.add_artist(kpt_text)

    plt.show()
