"""Fixed TAL+BCE+CIoU+DFL path from the supplied fork; AGPL-3.0.

Unused auxiliary heads, alternate IoU losses, segmentation, pose and OBB paths
have been removed. This is the active default loss path, not a new loss claim.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .assigner import TaskAlignedAssigner
from .geometry import bbox_iou, bbox2dist, dist2bbox, make_anchors, xywh2xyxy


class DFLoss(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist, target):
        target = target.clamp(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (F.cross_entropy(pred_dist, tl.reshape(-1), reduction="none").view(tl.shape) * wl
                + F.cross_entropy(pred_dist, tr.reshape(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.dfl_loss = DFLoss(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchors, target_bboxes, scores, score_sum, fg):
        weight = scores.sum(-1)[fg].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg], target_bboxes[fg], xywh=False, CIoU=True)
        loss_iou = ((1 - iou) * weight).sum() / score_sum
        # Retain the source fork's bbox2dist upper bound and DFL clamp.
        ltrb = bbox2dist(anchors, target_bboxes, self.reg_max - 1)
        loss_dfl = (self.dfl_loss(pred_dist[fg].reshape(-1, self.reg_max), ltrb[fg]) * weight).sum() / score_sum
        return loss_iou, loss_dfl


class DetectionLoss:
    def __init__(self, model, box=7.5, cls=0.5, dfl=1.5):
        self.device = next(model.parameters()).device
        head = model.model[-1]
        self.stride, self.nc, self.reg_max = head.stride, head.nc, head.reg_max
        self.no = self.nc + 4 * self.reg_max
        self.gains = (box, cls, dfl)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.reg_max).to(self.device)
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)

    def __call__(self, predictions, batch):
        feats = predictions[1] if isinstance(predictions, tuple) else predictions
        bs = feats[0].shape[0]
        dist, scores = torch.cat([p.view(bs, self.no, -1) for p in feats], 2).split((4*self.reg_max, self.nc), 1)
        dist, scores = dist.permute(0, 2, 1).contiguous(), scores.permute(0, 2, 1).contiguous()
        anchors, strides = make_anchors(feats, self.stride, 0.5)
        image_hw = torch.tensor(feats[0].shape[2:], dtype=scores.dtype, device=self.device) * self.stride[0]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1).to(self.device)
        counts = torch.bincount(targets[:, 0].long(), minlength=bs)
        padded = torch.zeros(bs, int(counts.max()), 5, device=self.device)
        for i in range(bs):
            t = targets[targets[:, 0] == i, 1:]
            padded[i, :len(t)] = t
        padded[..., 1:5] = xywh2xyxy(padded[..., 1:5] * image_hw[[1, 0, 1, 0]])
        gt_cls, gt_boxes = padded.split((1, 4), 2)
        gt_mask = gt_boxes.sum(-1, keepdim=True) > 0
        decoded = dist2bbox(dist.view(bs, -1, 4, self.reg_max).softmax(-1).matmul(self.proj.to(dist.dtype)), anchors, xywh=False)
        _, target_boxes, target_scores, fg, _ = self.assigner(scores.detach().sigmoid(),
            (decoded.detach()*strides).to(gt_boxes.dtype), anchors*strides, gt_cls, gt_boxes, gt_mask)
        total = target_scores.sum().clamp(min=1)
        cls_loss = self.bce(scores, target_scores.to(scores.dtype)).sum() / total
        # Keep a zero-valued graph connection for batches with no positive targets.
        box_loss, dfl_loss = dist.sum()*0, dist.sum()*0
        if fg.any():
            box_loss, dfl_loss = self.bbox_loss(dist, decoded, anchors, target_boxes/strides, target_scores, total, fg)
        components = torch.stack((box_loss*self.gains[0], cls_loss*self.gains[1], dfl_loss*self.gains[2]))
        return components.sum()*bs, components.detach()
