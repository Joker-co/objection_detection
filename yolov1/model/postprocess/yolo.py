import torch
import torch.nn as nn

class Yolov1PostProcess(nn.Module):
    def __init__(self, num_classes, stride, train=True):
        super(Yolov1PostProcess, self).__init__()
        self.num_classes = num_classes
        self.train = train
        self.stride = stride

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

    def generate_target(self, gt, padded_w, padded_h):
        xmin, ymin, xmax, ymax = gt[:-1]

    def get_targets(self, gt_bboxes, image_infos, stride):
        padded_h, padded_w = image_infos[0][-2:]
        B = len(gt_bboxes)

        for b_idx in range(B):
            b_gts = gt_bboxes[b_idx]
            for gt in b_gts:
                 label = int(gt[-1] - 1)
                 target = self.generate_target(gt, padded_w, padded_h)
    
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