import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import logging
logging.basicConfig(level=logging.INFO)

from yolov1.dataset.datasets import COCODataset
from yolov1.dataset.dataloader import HSampler, HBatchSampler, HDataLoader
from yolov1.model.backbone.resnet import resnet18
from yolov1.model.neck.CSPneck import CSPNeck
from yolov1.model.head.yolo import Yolov1Head
from yolov1.model.postprocess.yolo import Yolov1PostProcess
from yolov1.scheduler.lr_scheduler import HCombineLR

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed ** 2)
    torch.manual_seed(seed ** 3)
    torch.cuda.manual_seed(seed ** 4)

class Model(nn.Module):
    def __init__(self, num_classes, stride,
                 device=torch.device("cuda"), w_obj=5.0, w_noobj=1.0,
                 score_thresh=0.01, nms_thresh=0.5):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.stride = stride

        self.backbone = resnet18()
        self.neck = CSPNeck(512, 256)
        self.head = Yolov1Head(512, num_classes)
        self.postprocess = Yolov1PostProcess(num_classes,
                                             stride,
                                             device=device,
                                             w_obj=w_obj,
                                             w_noobj=w_noobj,
                                             score_thresh=score_thresh,
                                             nms_thresh=nms_thresh)

    def forward(self, input):
        input = copy.deepcopy(input)
        # backbone
        output = self.backbone(input)
        # neck
        output = self.neck(output)
        # head
        output = self.head(output)
        # postprocess
        output = self.postprocess(output)
        return output

def main():
    # set random seed
    seed = 131
    set_seed(seed)

    train_scales = [416, 416]
    batch_size = 1 # 32
    num_workers = 0 
    num_classes = 80
    lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-4 
    milestones = [60, 90, 160]
    warm_epoch = 2
    max_epoch = 160
    # build train data
    # train_meta = '/mnt/lustre/share/DSK/datasets/mscoco2017/annotations/instances_train2017.json'
    train_meta = 'datasets/debug_coco.json'
    # train_image_dir = '/mnt/lustre/share/DSK/datasets/mscoco2017/train2017'
    train_image_dir = 'datasets/debug_imgs'
    # pretrained_model = '/mnt/lustre/share/DSK/model_zoo/pytorch/imagenet/resnet18-5c106cde.pth' 
    pretrained_model = '../resnet18-5c106cde.pth'
    device = torch.device("cuda")

    train_dataset = COCODataset(train_meta, train_image_dir, scales=[train_scales[0]], max_scale=train_scales[1])
    # build train dataloader
    # build sampler
    train_sampler = HSampler(train_dataset)
    train_batch_sampler = HBatchSampler(train_sampler, batch_size=batch_size)
    # build dataloader
    train_dataloader = HDataLoader(dataset=train_dataset,
                                   batch_sampler=train_batch_sampler,
                                   num_workers=num_workers)
    iter_train_loader = iter(train_dataloader)

    # build model
    model = Model(num_classes, stride=32)
    # convert model to cuda & train mode
    model.to(device) # .train()
    model.train()
    
    # load pretrain
    cur_device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_model, map_location=lambda storage, loc: storage.cuda(cur_device))
    model.load_state_dict(pretrained_dict, strict=False)

    # build optimizer and lr_scheduler
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          momentum=momentum,
                          weight_decay=weight_decay)
    lr_scheduler = HCombineLR(epoch_size=len(train_dataloader),
                              optimizer=optimizer,
                              milestones=milestones,
                              warm_epoch=warm_epoch)
    
    epoch_size = len(train_dataloader)
    start_iters, max_iters = 0, max_epoch * len(train_dataloader)

    for iter_idx in range(start_iters, max_iters):
        try:
            batch = next(iter_train_loader)
        except StopIteration:
            print('update dataloader')
            epoch_idx = iter_idx // epoch_size
            train_dataloader.batch_sampler.sampler.set_seed(epoch_idx)
            test_batch_sampler = iter(train_batch_sampler)

        # forward
        output = model(batch)
        total_loss = 0
        for key in output:
            if 'loss' in key:
                total_loss += output[key]

        # update
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        # train logging
        train_meta = '[Epoch {}/{}][Iter {}/{}] LR: {}'.format(iter_idx // epoch_size,
                                                               max_epoch,
                                                               iter_idx % epoch_size + 1,
                                                               epoch_size,
                                                               round(lr_scheduler.get_lr()[0], 4))
        logging.info(train_meta)
        loss_meta = '[Loss] obj_loss: {} cls_loss: {} loc_loss: {} total_loss: {}'.format(round(output['obj_loss'].item(), 2),
                                                                                          round(output['cls_loss'].item(), 2),
                                                                                          round(output['loc_loss'].item(), 2),
                                                                                          round(output['total_loss'].item(), 2))
        logging.info(loss_meta)

if __name__ == "__main__":
    main()
