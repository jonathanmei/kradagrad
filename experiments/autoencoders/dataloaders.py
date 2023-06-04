import torch
import torchvision
import torchvision.transforms as transforms

from datasets import CURVESDataset, FACESDataset


def get_dataloaders(dataset, batch_size=100, root=None):
    if dataset == "mnist":
        train_set = torchvision.datasets.MNIST(
            root=root or "./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        test_set = torchvision.datasets.MNIST(
            root=root or "./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

    elif dataset == "faces":
        train_set = FACESDataset(
            root=root or "./data",
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=0.0, std=1.0)]
            ),
        )
        train_set, test_set = torch.utils.data.random_split(train_set, [103500, 62100])

    elif dataset == "curves":
        train_set = CURVESDataset(
            root=root or "./data",
            train=True,
        )

        test_set = CURVESDataset(
            root=root or "./data",
            train=False,
        )
    else:
        raise ValueError(f"Dataset {dataset} is not supported")

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader
