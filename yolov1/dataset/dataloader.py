import math
import torch
import torch.utils.data as data
from yolov1.dataset.datasets import COCODataset
from yolov1.dataset.transforms import Pad

try:
    from utils import get_world_size
    from utils import get_rank
except:
    get_world_size = None
    get_rank = None

class HSampler(data.Sampler):
    def __init__(self, dataset):
        super(HSampler, self).__init__(dataset)
        self.dataset = dataset
        self.epoch = 0
        
        try:
            self.num_device = get_world_size()
            self.rank = get_rank()
        except:
            self.num_device = 1
            self.rank = 0
        self.num_sampler = int(math.ceil(len(dataset) / self.num_device))
        self.total_size = self.num_sampler * self.num_device

    def __iter__(self):
        # shuffle dataset ids
        g = torch.Generator()
        g.manual_seed(self.epoch)
        shuffle_ids = list(torch.randperm(len(self.dataset), generator=g))

        # align shuffle ids into total_size
        num_extra = self.total_size - len(shuffle_ids)
        shuffle_ids = shuffle_ids + shuffle_ids[:num_extra]
        assert len(shuffle_ids) == self.total_size, "Dataset not be padded."

        # sample dataset for current device
        offset = self.rank * self.num_sampler
        sub_ids = shuffle_ids[offset:(offset + self.num_sampler)]
        return iter(sub_ids)

    def __len__(self):
        return self.num_sampler

    def set_seed(self, epoch):
        self.epoch = epoch

class HBatchSampler(data.BatchSampler):
    def __init__(self, sampler, batch_size, drop_last=False):
        super(HBatchSampler, self).__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        batches, batch = [], []
        for idx in self.sampler:
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []
            batch.append(idx)
        if len(batch) > 0 and not self.drop_last:
            batches.append(batch)
        return iter(batches)

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size) // self.batch_size

class HDataLoader(data.DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 batch_sampler=None,
                 num_workers=1,
                 sampler=None,
                 shuffle=False,
                 pin_memory=False,
                 drop_last=False,
                 pad_value=0,
                 alignment=32):
        super(HDataLoader, self).__init__(
            dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, self._collate_fn,
            pin_memory, drop_last)
        self.pad = Pad(pad_value, alignment)

    def _collate_fn(self, batch):
        images, gt_bboxes, image_infos = [], [], []
        for item in batch:
            images.append(item['image'])
            gt_bboxes.append(item['gt_bboxes'])
            image_infos.append(item['image_info'])
        
        # padded all images in batch
        images = self.pad(images, image_infos).cuda()
        _, _, padded_h, padded_w = images.shape
        for idx in range(len(image_infos)):
            image_infos[idx].append(padded_h)
            image_infos[idx].append(padded_w)
        return {
                'images': images,
                'gt_bboxes': gt_bboxes,
                'image_infos': image_infos
                }
    
    def __len__(self):
        return len(self.batch_sampler)

if __name__ == "__main__":
    dataset = COCODataset('debug_coco.json', 'debug_imgs')
    sampler = HSampler(dataset)
    iter_sampler = iter(sampler)
    batch_sampler = HBatchSampler(sampler, 8)
    print('dataset {}'.format(len(dataset)))
    print('sampler {} num_sampler {}'.format(sampler.dataset, sampler.num_sampler))
    for sid in sampler:
        print('sid {}'.format(sid))
    print('batch sampler {}'.format(len(batch_sampler)))
    for batch_ids in batch_sampler:
        print('batch_ids {}'.format(batch_ids))
    dataloader = HDataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)
    test_loader = iter(dataloader)
    test_batch_sampler = iter(batch_sampler)
    count = 1
    while count < 100:
        # next(test_batch_sampler)
        try:
            print(next(test_batch_sampler))
        except StopIteration:
            print('update batch_sampler')
            batch_sampler.sampler.set_seed(count)
            test_batch_sampler = iter(batch_sampler)
        print('dataloader count', count)
        count += 1
    print('length of batch sampler', len(dataloader.batch_sampler))