from torch.cuda import device_of
import warnings
import math
import os
import time
import torch
import urllib.request
import numpy as np
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

## Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms

from vit_solution import VisionTransformer

from utils.torch_utils import seed_experiment, to_device
#from utils.data_utils import save_logs

"""
# Configs to run

1. python run_exp.py --model vit --layers 2 --batch_size 128 --epochs 10 --optimizer adam
2. python run_exp.py --model vit --layers 2 --batch_size 128 --epochs 10 --optimizer adamw
3. python run_exp.py --model vit --layers 2 --batch_size 128 --epochs 10 --optimizer sgd
4. python run_exp.py --model vit --layers 2 --batch_size 128 --epochs 10 --optimizer momentum

5. python run_exp.py --model vit --layers 4 --batch_size 128 --epochs 10 --optimizer adamw
6. python run_exp.py --model vit --layers 6 --batch_size 128 --epochs 10 --optimizer adamw 
7. python run_exp.py --model vit --layers 6 --batch_size 128  --epochs 10 --optimizer adamw --block postnorm

"""


def train(epoch, model, dataloader, optimizer, args):

    model.train()

    total_iters = 0
    epoch_accuracy=0
    epoch_loss=0
    start_time = time.time()

    for idx, batch in enumerate(
        tqdm(
            dataloader, desc="Epoch {0}".format(epoch), disable=(not args.progress_bar)
        )
    ):
    
        batch = to_device(batch, args.device)
        optimizer.zero_grad()
        
        imgs, labels = batch
        logits = model(imgs)
        

        loss = model.loss(logits, labels)
        #print(loss)
        #losses.append(loss.item())
        acc = (logits.argmax(dim=1) == labels).float().mean()

        loss.backward()
        optimizer.step()
        epoch_accuracy += acc.item() / len(dataloader)
        epoch_loss += loss.item() / len(dataloader)
        total_iters += 1

        if idx % args.print_every == 0:
            tqdm.write(f"[TRAIN] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}")

    
    #print(epoch_loss)
    

    tqdm.write(f"== [TRAIN] Epoch: {epoch}, Accuracy: {epoch_accuracy:.3f} ==>")

    return epoch_loss, epoch_accuracy, time.time() - start_time


def evaluate(epoch, model, dataloader, args, mode="val"):
    model.eval()
    epoch_accuracy=0
    epoch_loss=0
    total_loss = 0.0
    total_iters = 0

    start_time = time.time()

    with torch.no_grad():
        for idx, batch in enumerate(
            tqdm(dataloader, desc="Evaluation", disable=(not args.progress_bar))
        ):
            batch = to_device(batch, args.device)

            imgs, labels = batch
            logits = model(imgs)
            

            loss = model.loss(logits, labels)
            acc = (logits.argmax(dim=1) == labels).float().mean()

            epoch_accuracy += acc.item() / len(dataloader)
            epoch_loss += loss.item() / len(dataloader)
            total_iters += 1

            if idx % args.print_every == 0:
                tqdm.write(
                    f"[{mode.upper()}] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}"
                )

    

        tqdm.write(
            f"=== [{mode.upper()}] Epoch: {epoch}, Iter: {idx}, Accuracy: {epoch_accuracy:.3f} ===>"
        )

    return epoch_loss, epoch_accuracy, time.time() - start_time


def main(args):
    # Seed the experiment, for repeatability
    seed_experiment(args.seed)

    test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                     ])
    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                        ])
    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    train_dataset = CIFAR10(root='./data', train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root='./data', train=True, transform=test_transform, download=True)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(args.seed))
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000], generator=torch.Generator().manual_seed(args.seed))
    # Loading the test set
    test_set = CIFAR10(root='./data', train=False, transform=test_transform, download=True)

    # We define a set of data loaders that we can use for various purposes later.
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    valid_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)

    # Model
    if args.model == "vit":
        model = VisionTransformer(
            num_layers=args.layers, block=args.block)
    else:
        raise ValueError("Unknown model {0}".format(args.model))
    model.to(args.device)

    # Optimizer
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "momentum":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    print(
        f"Initialized {args.model.upper()} model with {sum(p.numel() for p in model.parameters())} "
        f"total parameters, of which {sum(p.numel() for p in model.parameters() if p.requires_grad)} are learnable."
    )

    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []
    train_times, valid_times = [], []
    for epoch in range(10):

        tqdm.write(f"====== Epoch {epoch} ======>")

        loss, acc, wall_time = train(epoch, model, train_dataloader, optimizer,args)
        train_losses.append(loss)
        train_accs.append(acc)
        train_times.append(wall_time)

        loss, acc, wall_time = evaluate(epoch, model, valid_dataloader,args)
        valid_losses.append(loss)
        valid_accs.append(acc)
        valid_times.append(wall_time)

    test_loss, test_acc, test_time = evaluate(
        epoch, model, test_dataloader, args, mode="test"
    )

    print(f"===== Best validation Accuracy: {max(valid_accs):.3f} =====>")

    return (
        train_losses,
        train_accs,
        train_times,
        valid_losses,
        valid_accs,
        valid_times,
        test_loss,
        test_acc,
        test_time,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run an experiment for assignment 2.")

    data = parser.add_argument_group("Data")
    
    data.add_argument(
        "--batch_size", type=int, default=128, help="batch size (default: %(default)s)."
    )

    model = parser.add_argument_group("Model")
    model.add_argument(
        "--model",
        type=str,
        choices=["vit"],
        default="vit",
        help="name of the model to run (default: %(default)s).",
    )
    
    model.add_argument(
        "--layers",
        type=int,
        default=2,
        help="number of layers in the model (default: %(default)s).",
    )
    model.add_argument(
        "--block",
        type=str,
        default='prenorm',
        help="number of layers in the model (default: %(default)s).",
    )

    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="number of epochs for training (default: %(default)s).",
    )
    optimization.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["sgd", "momentum", "adam", "adamw"],
        help="choice of optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="learning rate for Adam optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum for SGD optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="weight decay (default: %(default)s).",
    )

    exp = parser.add_argument_group("Experiment config")
    exp.add_argument(
        "--exp_id",
        type=str,
        default="debug",
        help="unique experiment identifier (default: %(default)s).",
    )
    exp.add_argument(
        "--log",
        action="store_true",
        help="whether or not to log data from the experiment.",
    )
    exp.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="directory to log results to (default: %(default)s).",
    )
    exp.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for repeatability (default: %(default)s).",
    )

    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="number of processes to use for data loading (default: %(default)s).",
    )
    misc.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="device to store tensors on (default: %(default)s).",
    )
    misc.add_argument(
        "--progress_bar", action="store_true", help="show tqdm progress bar."
    )
    misc.add_argument(
        "--print_every",
        type=int,
        default=100,
        help="number of minibatches after which to print loss (default: %(default)s).",
    )

    args = parser.parse_args()

    # Check for the device
    if (args.device == "cuda") and not torch.cuda.is_available():
        warnings.warn(
            "CUDA is not available, make that your environment is "
            "running on GPU (e.g. in the Notebook Settings in Google Colab). "
            'Forcing device="cpu".'
        )
        args.device = "cpu"

    if args.device == "cpu":
        warnings.warn(
            "You are about to run on CPU, and might run out of memory "
            "shortly. You can try setting batch_size=1 to reduce memory usage."
        )

    logs = main(args)
    #Reuse the save logs function in utils to your needs if needed.
