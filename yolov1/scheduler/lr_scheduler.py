from torch.optim.lr_scheduler import MultiStepLR

class HMultiStepLR(MultiStepLR):
    def __init__(self, epoch_size, optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False):
        milestones = [miles * epoch_size for miles in milestones]
        super(HMultiStepLR, self).__init__(optimizer, milestones, gamma, last_epoch, verbose)

class HWarmupLR(object):
    def __init__(self, warm_epoch, epoch_size):
        self.warm_iters = warm_epoch * epoch_size

    def get_lr(self, base_lrs, last_epoch):
        return [base_lr * pow((last_epoch / self.warm_iters), 4) for base_lr in base_lrs]

class HCombineLR(HMultiStepLR):
    def __init__(self,
                 epoch_size,
                 optimizer,
                 milestones,
                 warm_epoch,
                 gamma=0.1,
                 last_epoch=-1,
                 verbose=False):
        super(HCombineLR, self).__init__(epoch_size, optimizer, milestones, gamma, last_epoch, verbose)
        self.warmup_lr_scheduler = HWarmupLR(warm_epoch, epoch_size)
        self.base_lrs = [param_group['lr'] for param_group in optimizer.param_groups]

    def get_lr(self):
        if self.last_epoch < self.warmup_lr_scheduler.warm_iters:
            return self.warmup_lr_scheduler.get_lr(self.base_lrs, self.last_epoch)
        else:
            return super(HCombineLR, self).get_lr()