import torch
import torch.nn.functional as F


def kpts_loss(output, target):
    pred_kpts = output['pred_kpts']
    pred_visibility = output['pred_visibility']

    tgt_raw = torch.cat([v['keypoints'] for v in target])
    tgt_kpts = tgt_raw[:, :, :2]
    tgt_visibility = tgt_raw[:, :, 2:].flatten(1)

    loss_kpts = F.l1_loss(pred_kpts, tgt_kpts, reduction='mean')
    loss_visibility = F.binary_cross_entropy(pred_visibility, tgt_visibility)

    loss = loss_kpts + loss_visibility

    return loss
