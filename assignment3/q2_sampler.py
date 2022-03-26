import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler


def svhn_sampler(root, train_batch_size, test_batch_size, valid_split=0):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    transform = transforms.Compose((
        transforms.ToTensor(),
        normalize))
    train = datasets.SVHN(root, split='train', download=True, transform=transform)
    valid = datasets.SVHN(root, split='train', download=True, transform=transform)
    test = datasets.SVHN(root, split='test', download=True, transform=transform)

    idxes = np.arange(len(train))
    split = int(valid_split * len(idxes))
    train_sampler = SubsetRandomSampler(idxes[split:])
    valid_sampler = SubsetRandomSampler(idxes[:split])
    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               sampler=train_sampler, num_workers=4, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=train_batch_size,
                                               sampler=valid_sampler, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size, num_workers=4)

    return train_loader, valid_loader, test_loader,
