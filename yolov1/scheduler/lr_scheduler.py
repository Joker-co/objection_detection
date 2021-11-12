from torch.optim.lr_scheduler import MultiStepLR

class HMultiStepLR(MultiStepLR):
    def __init__(self, epoch_size, optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False):
        milestones = [miles * epoch_size for miles in milestones]
        super(HMultiStepLR, self).__init__(optimizer, milestones, gamma, last_epoch, verbose)

class HWarmupLR(object):
    