import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

class MSELoss(_Loss):
    def __init__(self, w_obj, w_noobj, reduction='none'):
        super(MSELoss, self).__init__(reduction=reduction)
        self.w_obj = w_obj
        self.w_noobj = w_noobj
        self.reduction = reduction

    def forward(self, input, target):
        assert self.reduction in ['mean'], 'Only support mean mode.'
        pos_idx = (target == 1.0).float()
        neg_idx = (target == 0.0).float()
        pos_loss = pos_idx * (inputs - targets) ** 2
        neg_loss = neg_idx * (inputs) ** 2
        if self.reduction == 'mean':
            pos_loss = torch.mean(torch.sum(pos_loss, 1))
            neg_loss = torch.mean(torch.sum(neg_loss, 1))
        
        return self.w_obj * pos_loss + self.w_noobj * neg_loss

class Yolov1PostProcess(nn.Module):
    def __init__(self, num_classes, stride, train=True, device=torch.device("cuda"), w_obj=5.0, w_noobj=1.0,
                 score_thresh=0.01, nms_thresh=0.5):
        super(Yolov1PostProcess, self).__init__()
        self.num_classes = num_classes
        self.train = train
        self.stride = stride
        self.device = device
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

        # build loss
        self.obj_loss = MSELoss(w_obj, w_noobj, reduction='mean')
        self.cls_loss = nn.CrossEntropyLoss(reduction='none')
        self.loc_center_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.loc_border_loss = nn.MSELoss(reduction='none')

    def filter(self, gt_bboxes, image_infos, min_size=1):
        padded_h, padded_w = image_infos[0][-2:]
        valid_gt_bboxes = []
        for gts in gt_bboxes:
            v_gts = gts.clone()
            v_gts[:, 0] = torch.clamp(v_gts[:, 0], min=0, max=padded_w)
            v_gts[:, 1] = torch.clamp(v_gts[:, 1], min=0, max=padded_h)
            v_gts[:, 2] = torch.clamp(v_gts[:, 2], min=0, max=padded_w)
            v_gts[:, 3] = torch.clamp(v_gts[:, 3], min=0, max=padded_h)

            bbox_w = v_gts[:, 2] - v_gts[:, 0]
            bbox_h = v_gts[:, 3] - v_gts[:, 1]
            valid_mask = (bbox_w >= min_size) and (bbox_h >= min_size)
            mask_v_gts = v_gts[valid_mask]
            valid_gt_bboxes.append(v_gts)
        return valid_gt_bboxes

    def generate_target(self, gt, padded_w, padded_h, stride):
        xmin, ymin, xmax, ymax = gt[:-1]
        # cal gt center, width, and height
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        b_w = xmax - xmin
        b_h = ymax - ymin

        # map center to grid cell
        map_cx = cx / stride
        map_cy = cy / stride
        grid_cx = int(map_cx)
        grid_cy = int(map_cy)
        # cal location target
        tx = map_cx - grid_cx
        ty = map_cy - grid_cy
        tw = np.log(b_w)
        th = np.log(b_h)
        weight = 2.0 - (b_w / padded_w) * (b_h / padded_h)

        return grid_cx, grid_cy, tx, ty, tw, th, weight

    def get_targets(self, gt_bboxes, image_infos, stride):
        padded_h, padded_w = image_infos[0][-2:]
        B = len(gt_bboxes)

        # generate target tensor
        ws = w // stride
        hs = h // stride
        target_tensor = torch.zeros([B, hs, ws, 7])

        for b_idx in range(B):
            b_gts = gt_bboxes[b_idx]
            for gt in b_gts:
                 label = int(gt[-1] - 1)
                 target = self.generate_target(gt, padded_w, padded_h, stride)
                 grid_cx, grid_cy, tx, ty, tw, th, weight = target

                 if grid_cx < target_tensor.shape[2] and grid_cy < target_tensor.shape[1]:
                     # for objectness
                     target_tensor[b_idx, grid_cy, grid_cx, 0] = 1.0
                     # for class
                     target_tensor[b_idx, grid_cy, grid_cx, 1] = label
                     # for location
                     target_tensor[b_idx, grid_cy, grid_cx, 2:6] = torch.tensor([tx, ty, tw, th])
                     target_tensor[b_idx, grid_cy, grid_cx, 6] = weight
        
        targets = target_tensor.view(B, -1, 7).float().to(self.device)
        return targets

    def get_loss(self, obj_pred, cls_pred, loc_pred, targets):
        # objectness
        obj_pred = obj_pred.sigmoid()
        obj_target = targets[:, :, 0].float()
        obj_loss = self.obj_loss(obj_pred, obj_target)

        # class
        cls_pred = cls_pred.permute(0, 2, 1)
        cls_target = targets[:, :, 1].long()
        cls_loss = self.cls_loss(cls_pred, cls_target) * obj_target
        cls_loss = torch.mean(torch.sum(cls_loss, 1))

        loc_weight = targets[:, :, 6]
        # loc center
        loc_center_pred = loc_pred[:, :, :2]
        loc_center_target = targets[:, :, 2:4].float()
        loc_center_loss = self.loc_center_loss(loc_center_pred, loc_center_target)
        loc_center_loss = torch.sum(loc_center_loss, 2) * loc_weight * obj_target
        loc_center_loss = torch.mean(torch.sum(loc_center_loss, 1))
        # loc border
        loc_border_pred = loc_pred[:, :, 2:]
        loc_border_target = targets[:, :, 4:6].float()
        loc_border_loss = self.loc_border_loss(loc_border_pred, loc_border_target)
        loc_border_loss = torch.sum(loc_border_loss, 2) * loc_weight * obj_target
        loc_border_loss = torch.mean(torch.sum(loc_border_loss, 1))

        total_loss = obj_loss + cls_loss + loc_center_loss + loc_border_loss

        return {
            'obj_loss': obj_loss,
            'cls_loss': cls_loss,
            'loc_loss': loc_center_loss + loc_border_loss,
            'total_loss': total_loss
        }

    def decode_bbox(self, loc_pred, grids):
        loc_pred = loc_pred.clone()
        dt_bboxes = torch.zeros_like(loc_pred)
        loc_pred[:, :, :2] = loc_pred[:, :, :2].sigmoid() + grids
        loc_pred[:, :, 2:] = torch.exp(loc_pred[:, :, 2:])

        dt_bboxes[:, :, 0] = loc_pred[:, :, 0] * self.stride - loc_pred[:, :, 2] / 2
        dt_bboxes[:, :, 1] = loc_pred[:, :, 1] * self.stride - loc_pred[:, :, 3] / 2
        dt_bboxes[:, :, 2] = loc_pred[:, :, 2] * self.stride + loc_pred[:, :, 2] / 2
        dt_bboxes[:, :, 3] = loc_pred[:, :, 3] * self.stride + loc_pred[:, :, 3] / 2

        return dt_bboxes
    
    def generate_grids(self, image_infos):
        padded_h, padded_w = image_infos[0][-2:]
        # downsample
        feat_h, feat_w = padded_h // self.stride, padded_w // self.stride
        grid_cy, grid_cx = torch.meshgrid([torch.arange(feat_h), torch.arange(feat_w)])
        grids = torch.stack([grid_cx, grid_cy], dim=-1).float()

        return grids

    def nms(self, bboxes, scores):
        order = scores.argsort(dim=0)[::-1]
        # calculate bboxes area
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        keep = []
        while order.numel() > 0:
            # max score box id
            idx = order[0]
            keep.append(idx)
            xx1 = torch.maximum(x1[idx], x1[order[1:]])
            yy1 = torch.maximum(y1[idx], y1[order[1:]])
            xx2 = torch.minimum(x2[idx], x2[order[1:]])
            yy2 = torch.minimum(y2[idx], y2[order[1:]])

            w = torch.maximum(0, xx2 - xx1)
            h = torch.maximum(0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[idx] + areas[order[1:]] - inter)
            inds = torch.nonzero(ovr <= self.nms_thresh).view(-1)
            order = order[inds + 1]
        return keep

    def predict(self, dt_bboxes, scores):
        """
        dt_bboxes: (B, K, 4)
        scores: (B, K, num_classes)
        """
        dt_bboxes = dt_bboxes.clone()
        scores = scores.clone()
        B, _, _ = dt_bboxes.shape()

        batch_dets = []
        for b_idx in range(B):
            # (K, 4)
            b_bboxes = dt_bboxes[b_idx]
            # (K, num_classes)
            b_scores = scores[b_idx]
            # select max class score
            cls_inds = torch.argmax(b_scores, axis=1)
            b_scores = b_scores[torch.arange(b_scores.shape[0]), cls_inds]
            # filter by threshold
            keep = b_scores >= self.score_thresh
            b_scores = b_scores[keep]
            b_bboxes = b_bboxes[keep]
            cls_inds = cls_inds[keep]

            # nms
            keep = torch.zeros(cls_inds.shape[0], dtype=torch.int8)
            for cls in range(self.num_classes):
                inds = torch.nonzero(cls_inds == cls).view(-1)
                if inds.numel() == 0:
                    continue
                cls_bboxes = b_bboxes[inds]
                cls_scores = b_scores[inds]
                keep_nms = self.nms(cls_bboxes, cls_scores)
                keep_nms = torch.tensor(keep_nms)
                keep[inds[keep_nms]] = 1
            
            det_bboxes = b_bboxes[keep > 0]
            det_scores = b_scores[keep > 0]
            det_labels = cls_inds[keep > 0]
            batch_dets.append([det_bboxes, det_scores, det_labels])
        
        return batch_dets

    def forward(self, x):
        preds = x['feats']
        gt_bboxes = x['gt_bboxes']
        image_infos = x['image_infos']
        padded_h, padded_w = image_infos[0][-2:]

        # filter dirty gts
        valid_gt_bboxes = self.filter(gt_bboxes, image_infos)

        B, C, _, _ = preds.shape
        # permute
        preds = preds.view(B, C, -1).permute(0, 2, 1)
        B, K, C = preds.size()

        # B, K, 1
        obj_pred = preds[:, :, 0]
        # B, K, num_classes
        cls_pred = preds[:, :, 1:(1 + self.num_classes)]
        # B, K, 4
        loc_pred = preds[:, :, -4:]

        if self.train:
            targets = self.get_targets(valid_gt_bboxes, image_infos, self.stride)
            losses = self.get_loss(obj_pred, cls_pred, loc_pred, targets)
            return losses
        else:
            # generate grids
            grids = self.generate_grids(image_infos)
            grids = grids.view(B, -1, 2).to(self.device)
            with torch.no_grad():
                # [B, K, 1]
                obj_pred = obj_pred.sigmoid()
                # decode bboxes
                dt_bboxes = self.decode_bbox(loc_pred, grids)
                dt_bboxes[:, :, 0] = torch.clamp(dt_bboxes[:, :, 0], min=0, max=padded_w)
                dt_bboxes[:, :, 1] = torch.clamp(dt_bboxes[:, :, 1], min=0, max=padded_h)
                dt_bboxes[:, :, 2] = torch.clamp(dt_bboxes[:, :, 2], min=0, max=padded_w)
                dt_bboxes[:, :, 3] = torch.clamp(dt_bboxes[:, :, 3], min=0, max=padded_h)
                scores = cls_pred.softmax(dim=2) * obj_pred

                dt_results = self.predict(dt_bboxes, scores)
            return dt_results