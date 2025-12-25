# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.yolo.utils.metrics import OKS_SIGMA
from ultralytics.yolo.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

from .metrics import bbox_iou
from .tal import bbox2dist

import numpy as np


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367."""

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.amp.autocast('cuda', enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') *
                    weight).mean(1).sum()
        return loss


# Losses
class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self, ):
        super().__init__()

    def forward(self, pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


class KeypointLoss(nn.Module):

    def __init__(self, sigmas) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0)) / (torch.sum(kpt_mask != 0) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2 * self.sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        return kpt_loss_factor * ((1 - torch.exp(-e)) * kpt_mask).mean()


loss_t = []

# Criterion class for computing Detection training losses
class v8DetectionLoss:

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device
        self.cube = h.cube
        self.dwa = h.dwa

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            if self.cube:
                out = torch.zeros(batch_size, 0, 13, device=self.device)
            else:
                out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            if self.cube:
                out = torch.zeros(batch_size, counts.max(), 13, device=self.device)
            else:
                out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
            if self.cube:
                out[..., 5:9] = xywh2xyxy(out[..., 5:9].mul_(scale_tensor))
                out[..., 9:13] = xywh2xyxy(out[..., 9:13].mul_(scale_tensor))
                # out[..., 13:17] = xywh2xyxy(out[..., 13:17].mul_(scale_tensor))
                # out[..., 17:21] = xywh2xyxy(out[..., 17:21].mul_(scale_tensor))
                # out[..., 21:25] = xywh2xyxy(out[..., 21:25].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)
    
    def clamp_number(self, num):
        a = torch.Tensor([1.0]).to(num.device)
        b = -1 * a
        return torch.max(torch.min(num, a), b)

    def dynamic_weight_average(self, loss_t):
        assert len(loss_t) == 5
        loss_t_2 = loss_t[3]
        loss_t_1 = loss_t[4]
        assert len(loss_t_1) == len(loss_t_2)
        w = [self.clamp_number(l_1 / l_2 - 1) for l_1, l_2 in zip(loss_t_1.clone(), loss_t_2.clone())]
        lamb = [(1 / (1 + np.exp(-5 * v.cpu().detach().numpy()))) for v in w]
        lamb_sum = sum(lamb)
        task_n = len(loss_t_1)
        weight = [task_n * l / lamb_sum for l in lamb]
        return weight

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        if self.cube:
            loss_box = torch.zeros(3, device=self.device)
            loss_dfl = torch.zeros(3, device=self.device)
            loss_cls = torch.zeros(3, device=self.device)
            if self.dwa:
                loss_123 = torch.zeros(3, device=self.device)
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds     #[0]:[8, 193, 80, 80]   [1]:[8, 193, 40, 40]    [2]:[8, 193, 20, 20]
        if self.cube:
            # pred_distri_t1, pred_scores_t1, pred_distri_t2, pred_scores_t2, pred_distri_t3, pred_scores_t3, pred_distri_t4, pred_scores_t4, pred_distri_t5, pred_scores_t5, pred_distri_t6, pred_scores_t6 = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            #     (self.reg_max * 4, self.nc, self.reg_max * 4, self.nc, self.reg_max * 4, self.nc, self.reg_max * 4, self.nc, self.reg_max * 4, self.nc, self.reg_max * 4, self.nc), 1)
            pred_distri_t1, pred_scores_t1, pred_distri_t2, pred_scores_t2, pred_distri_t3, pred_scores_t3 = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
                (self.reg_max * 4, self.nc, self.reg_max * 4, self.nc, self.reg_max * 4, self.nc), 1)
            pred_distri_t1 = pred_distri_t1.permute(0, 2, 1).contiguous()
            pred_scores_t1 = pred_scores_t1.permute(0, 2, 1).contiguous()
            pred_distri_t2 = pred_distri_t2.permute(0, 2, 1).contiguous()
            pred_scores_t2 = pred_scores_t2.permute(0, 2, 1).contiguous()
            pred_distri_t3 = pred_distri_t3.permute(0, 2, 1).contiguous()
            pred_scores_t3 = pred_scores_t3.permute(0, 2, 1).contiguous()
            # pred_distri_t4 = pred_distri_t4.permute(0, 2, 1).contiguous()
            # pred_scores_t4 = pred_scores_t4.permute(0, 2, 1).contiguous()
            # pred_distri_t5 = pred_distri_t5.permute(0, 2, 1).contiguous()
            # pred_scores_t5 = pred_scores_t5.permute(0, 2, 1).contiguous()
            # pred_distri_t6 = pred_distri_t6.permute(0, 2, 1).contiguous()
            # pred_scores_t6 = pred_scores_t6.permute(0, 2, 1).contiguous()
            dtype = pred_scores_t3.dtype
            batch_size = pred_scores_t3.shape[0]
        else:
            pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
                (self.reg_max * 4, self.nc), 1)

            pred_scores = pred_scores.permute(0, 2, 1).contiguous()
            pred_distri = pred_distri.permute(0, 2, 1).contiguous()

            dtype = pred_scores.dtype
            batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)    # [8400, 2] [8400, 1]

        # targets
        if self.cube:
            targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes_t1'], batch['bboxes_t2'], batch['bboxes_t3']), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])    # [8, 4, 13]
            gt_labels, gt_bboxes_t1, gt_bboxes_t2, gt_bboxes_t3 = targets.split((1, 4, 4, 4), 2)  # cls, xyxy
            mask_gt_t1 = gt_bboxes_t1.sum(2, keepdim=True).gt_(0)   # [8, 4, 1]
            mask_gt_t2 = gt_bboxes_t2.sum(2, keepdim=True).gt_(0)
            mask_gt_t3 = gt_bboxes_t3.sum(2, keepdim=True).gt_(0)
            # mask_gt_t4 = gt_bboxes_t4.sum(2, keepdim=True).gt_(0)
            # mask_gt_t5 = gt_bboxes_t5.sum(2, keepdim=True).gt_(0)
            # mask_gt_t6 = gt_bboxes_t6.sum(2, keepdim=True).gt_(0)
        else:
            targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        if self.cube:
            pred_bboxes_t1 = self.bbox_decode(anchor_points, pred_distri_t1)  # xyxy, (b, h*w, 4)
            pred_bboxes_t2 = self.bbox_decode(anchor_points, pred_distri_t2)  # xyxy, (b, h*w, 4)
            pred_bboxes_t3 = self.bbox_decode(anchor_points, pred_distri_t3)  # xyxy, (b, h*w, 4)
            # pred_bboxes_t4 = self.bbox_decode(anchor_points, pred_distri_t4)
            # pred_bboxes_t5 = self.bbox_decode(anchor_points, pred_distri_t5)
            # pred_bboxes_t6 = self.bbox_decode(anchor_points, pred_distri_t6)

            _, target_bboxes_t1, target_scores_t1, fg_mask_t1, _ = self.assigner(
                pred_scores_t1.detach().sigmoid(), (pred_bboxes_t1.detach() * stride_tensor).type(gt_bboxes_t1.dtype),
                anchor_points * stride_tensor, gt_labels, gt_bboxes_t1, mask_gt_t1)
        
            _, target_bboxes_t2, target_scores_t2, fg_mask_t2, _ = self.assigner(
                pred_scores_t2.detach().sigmoid(), (pred_bboxes_t2.detach() * stride_tensor).type(gt_bboxes_t2.dtype),
                anchor_points * stride_tensor, gt_labels, gt_bboxes_t2, mask_gt_t2)
        
            _, target_bboxes_t3, target_scores_t3, fg_mask_t3, _ = self.assigner(
                pred_scores_t3.detach().sigmoid(), (pred_bboxes_t3.detach() * stride_tensor).type(gt_bboxes_t3.dtype),
                anchor_points * stride_tensor, gt_labels, gt_bboxes_t3, mask_gt_t3)

            # _, target_bboxes_t4, target_scores_t4, fg_mask_t4, _ = self.assigner(
            #     pred_scores_t4.detach().sigmoid(), (pred_bboxes_t4.detach() * stride_tensor).type(gt_bboxes_t4.dtype),
            #     anchor_points * stride_tensor, gt_labels, gt_bboxes_t4, mask_gt_t4)

            # _, target_bboxes_t5, target_scores_t5, fg_mask_t5, _ = self.assigner(
            #     pred_scores_t5.detach().sigmoid(), (pred_bboxes_t5.detach() * stride_tensor).type(gt_bboxes_t5.dtype),
            #     anchor_points * stride_tensor, gt_labels, gt_bboxes_t5, mask_gt_t5)

            # _, target_bboxes_t6, target_scores_t6, fg_mask_t6, _ = self.assigner(
            #     pred_scores_t6.detach().sigmoid(), (pred_bboxes_t6.detach() * stride_tensor).type(gt_bboxes_t6.dtype),
            #     anchor_points * stride_tensor, gt_labels, gt_bboxes_t6, mask_gt_t6)

            target_scores_sum_t1 = max(target_scores_t1.sum(), 1)
            target_scores_sum_t2 = max(target_scores_t2.sum(), 1)
            target_scores_sum_t3 = max(target_scores_t3.sum(), 1)
            # target_scores_sum_t4 = max(target_scores_t4.sum(), 1)
            # target_scores_sum_t5 = max(target_scores_t5.sum(), 1)
            # target_scores_sum_t6 = max(target_scores_t6.sum(), 1)

            # cls loss
            loss_cls[0] = self.bce(pred_scores_t1, target_scores_t1.to(dtype)).sum() / target_scores_sum_t1
            loss_cls[1] = self.bce(pred_scores_t2, target_scores_t2.to(dtype)).sum() / target_scores_sum_t2
            loss_cls[2] = self.bce(pred_scores_t3, target_scores_t3.to(dtype)).sum() / target_scores_sum_t3
            # loss_cls[3] = self.bce(pred_scores_t4, target_scores_t4.to(dtype)).sum() / target_scores_sum_t4
            # loss_cls[4] = self.bce(pred_scores_t5, target_scores_t5.to(dtype)).sum() / target_scores_sum_t5
            # loss_cls[5] = self.bce(pred_scores_t6, target_scores_t6.to(dtype)).sum() / target_scores_sum_t6

            # bbox loss
            if fg_mask_t1.sum():
                target_bboxes_t1 /= stride_tensor
                loss_box[0], loss_dfl[0] = self.bbox_loss(pred_distri_t1, pred_bboxes_t1, anchor_points, target_bboxes_t1, target_scores_t1,
                                                target_scores_sum_t1, fg_mask_t1)
            
            if fg_mask_t2.sum():
                target_bboxes_t2 /= stride_tensor
                loss_box[1], loss_dfl[1] = self.bbox_loss(pred_distri_t2, pred_bboxes_t2, anchor_points, target_bboxes_t2, target_scores_t2,
                                                target_scores_sum_t2, fg_mask_t2)
            
            if fg_mask_t3.sum():
                target_bboxes_t3 /= stride_tensor
                loss_box[2], loss_dfl[2] = self.bbox_loss(pred_distri_t3, pred_bboxes_t3, anchor_points, target_bboxes_t3, target_scores_t3,
                                                target_scores_sum_t3, fg_mask_t3)

            # if fg_mask_t4.sum():
            #     target_bboxes_t4 /= stride_tensor
            #     loss_box[3], loss_dfl[3] = self.bbox_loss(pred_distri_t4, pred_bboxes_t4, anchor_points, target_bboxes_t4, target_scores_t4,
            #                                     target_scores_sum_t4, fg_mask_t4)

            # if fg_mask_t5.sum():
            #     target_bboxes_t5 /= stride_tensor
            #     loss_box[4], loss_dfl[4] = self.bbox_loss(pred_distri_t5, pred_bboxes_t5, anchor_points, target_bboxes_t5, target_scores_t5,
            #                                     target_scores_sum_t5, fg_mask_t5)
            
            # if fg_mask_t6.sum():
            #     target_bboxes_t6 /= stride_tensor
            #     loss_box[5], loss_dfl[5] = self.bbox_loss(pred_distri_t6, pred_bboxes_t6, anchor_points, target_bboxes_t6, target_scores_t6,
            #                                     target_scores_sum_t6, fg_mask_t6)

            loss_cls *= self.hyp.cls
            loss_box *= self.hyp.box
            loss_dfl *= self.hyp.dfl
            
            if self.dwa:
                for t_i in range(3):
                    loss_123[t_i] = loss_cls[t_i] + loss_box[t_i] + loss_dfl[t_i]

                global loss_t

                loss_t.append(loss_123)
                if len(loss_t) < 5:
                    weight = [1.0,1.0,1.0]
                    for t_i in range(3):
                        loss_cls[t_i] *= weight[t_i]
                        loss_box[t_i] *= weight[t_i]
                        loss_dfl[t_i] *= weight[t_i]
                else:
                    weight = self.dynamic_weight_average(loss_t)
                    loss_t.pop(0)
                    for t_i in range(3):
                        loss_cls[t_i] *= torch.tensor(weight[t_i][0])
                        loss_box[t_i] *= torch.tensor(weight[t_i][0])
                        loss_dfl[t_i] *= torch.tensor(weight[t_i][0])

            loss[0] = (loss_cls[0] + loss_cls[1] + loss_cls[2]) / 3
            loss[1] = (loss_box[0] + loss_box[1] + loss_box[2]) / 3
            loss[2] = (loss_dfl[0] + loss_dfl[1] + loss_dfl[2]) / 3
        else:
            pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

            _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
                pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

            target_scores_sum = max(target_scores.sum(), 1)

            # cls loss
            # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
            loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

            # bbox loss
            if fg_mask.sum():
                target_bboxes /= stride_tensor
                loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                                target_scores_sum, fg_mask)

            loss[0] *= self.hyp.box  # box gain
            loss[1] *= self.hyp.cls  # cls gain
            loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


# Criterion class for computing training losses
class v8SegmentationLoss(v8DetectionLoss):

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.nm = model.model[-1].nm  # number of masks
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch['batch_idx'].view(-1, 1)
            targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError('ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n'
                            "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                            "i.e. 'yolo train model=yolov8n-seg.pt data=coco128.yaml'.\nVerify your dataset is a "
                            "correctly formatted 'segment' dataset using 'data=coco128-seg.yaml' "
                            'as an example.\nSee https://docs.ultralytics.com/tasks/segment/ for help.') from e

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # bbox loss
            loss[0], loss[3] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                              target_scores, target_scores_sum, fg_mask)
            # masks loss
            masks = batch['masks'].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            for i in range(batch_size):
                if fg_mask[i].sum():
                    mask_idx = target_gt_idx[i][fg_mask[i]]
                    if self.overlap:
                        gt_mask = torch.where(masks[[i]] == (mask_idx + 1).view(-1, 1, 1), 1.0, 0.0)
                    else:
                        gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                    xyxyn = target_bboxes[i][fg_mask[i]] / imgsz[[1, 0, 1, 0]]
                    marea = xyxy2xywh(xyxyn)[:, 2:].prod(1)
                    mxyxy = xyxyn * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device)
                    loss[1] += self.single_mask_loss(gt_mask, pred_masks[i][fg_mask[i]], proto[i], mxyxy, marea)  # seg

                # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
                else:
                    loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box / batch_size  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        """Mask loss for one image."""
        pred_mask = (pred @ proto.view(self.nm, -1)).view(-1, *proto.shape[1:])  # (n, 32) @ (32,80,80) -> (n,80,80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()


# Criterion class for computing training losses
class v8PoseLoss(v8DetectionLoss):

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            for i in range(batch_size):
                if fg_mask[i].sum():
                    idx = target_gt_idx[i][fg_mask[i]]
                    gt_kpt = keypoints[batch_idx.view(-1) == i][idx]  # (n, 51)
                    gt_kpt[..., 0] /= stride_tensor[fg_mask[i]]
                    gt_kpt[..., 1] /= stride_tensor[fg_mask[i]]
                    area = xyxy2xywh(target_bboxes[i][fg_mask[i]])[:, 2:].prod(1, keepdim=True)
                    pred_kpt = pred_kpts[i][fg_mask[i]]
                    kpt_mask = gt_kpt[..., 2] != 0
                    loss[1] += self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss
                    # kpt_score loss
                    if pred_kpt.shape[-1] == 3:
                        loss[2] += self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose / batch_size  # pose gain
        loss[2] *= self.hyp.kobj / batch_size  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def kpts_decode(self, anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y


class v8ClassificationLoss:

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch['cls'], reduction='sum') / 64
        loss_items = loss.detach()
        return loss, loss_items
