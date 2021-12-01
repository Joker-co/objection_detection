from yolov1.dataset.datasets import COCODataset
from yolov1.dataset.dataloader import HSampler, HBatchSampler, HDataLoader

def main():
    # build train data
    train_meta = ''
    train_image_dir = ''
    train_dataset = COCODataset(meta_file, train_image_dir)
    # build train dataloader
    # build sampler
    train_sampler = HSampler(train_dataset)
    train_batch_sampler = HBatchSampler()

if __name__ == "__main__":
    main()