import os
import pickle
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class LocalCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.data = []
        self.labels = []
        with open(os.path.join(root, "batches.meta"), "rb") as f:
            meta = pickle.load(f, encoding="latin1")
        self.classes = meta["label_names"]
        if self.train:
            batch_files = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            batch_files = ["test_batch"]
        for fname in batch_files:
            path = os.path.join(root, fname)
            with open(path, "rb") as f:
                batch = pickle.load(f, encoding="latin1")
            self.data.extend(batch["data"])
            self.labels.extend(batch["labels"])
        self.data = [img.reshape(3, 32, 32).transpose(1, 2, 0) for img in self.data]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

class TwoCropTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform
    def __call__(self, x):
        view1 = self.base_transform(x)
        view2 = self.base_transform(x)
        return view1, view2

def get_simclr_augment():
    aug = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return aug

def get_eval_transform():
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return trans