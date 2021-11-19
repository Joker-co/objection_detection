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
    def __init__(self, num_classes, stride, train=True, device='cuda', w_obj=5.0, w_noobj=1.0):
        super(Yolov1PostProcess, self).__init__()
        self.num_classes = num_classes
        self.train = train
        self.stride = stride
        self.device = device

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
            'loc_center_loss': loc_center_loss,
            'loc_border_loss': loc_border_loss
        }
    
    def forward(self, x):
        preds = x['feats']
        gt_bboxes = x['gt_bboxes']
        image_infos = x['image_infos']

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