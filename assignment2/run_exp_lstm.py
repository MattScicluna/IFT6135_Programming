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

from lstm_solution import LSTM
from utils.wikitext2 import Wikitext2
from utils.torch_utils import seed_experiment, to_device
from utils.data_utils import save_logs



"""
# Configs to run

1. python run_exp.py --model lstm --layers 1 --batch_size 16 --log --epochs 10 --optimizer adam
2. python run_exp.py --model lstm --layers 1 --batch_size 16 --log --epochs 10 --optimizer adamw
3. python run_exp.py --model lstm --layers 1 --batch_size 16 --log --epochs 10 --optimizer sgd
4. python run_exp.py --model lstm --layers 1 --batch_size 16 --log --epochs 10 --optimizer momentum

5. python run_exp.py --model lstm --layers 2 --batch_size 16 --log --epochs 10 --optimizer adamw
6. python run_exp.py --model lstm --layers 4 --batch_size 16 --log --epochs 10 --optimizer adamw

"""


def train(epoch, model, dataloader, optimizer, args):

    model.train()

    losses = []
    total_iters = 0

    start_time = time.time()

    for idx, batch in enumerate(
        tqdm(
            dataloader, desc="Epoch {0}".format(epoch), disable=(not args.progress_bar)
        )
    ):
        batch = to_device(batch, args.device)
        optimizer.zero_grad()

        
        hidden_states = model.initial_states(batch["source"].shape[0])
        log_probas, _ = model(batch["source"], hidden_states)
        

        loss = model.loss(log_probas, batch["target"], batch["mask"])
        losses.append(loss.item() * batch["mask"].sum().item())

        loss.backward()
        optimizer.step()

        total_iters += 1

        if idx % args.print_every == 0:
            tqdm.write(f"[TRAIN] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}")

    mean_loss = np.mean(losses)
    mean_loss /= args.batch_size * dataloader.dataset.max_length

    perplexity = math.exp(mean_loss)

    tqdm.write(f"== [TRAIN] Epoch: {epoch}, Perplexity: {perplexity:.3f} ==>")

    return mean_loss, perplexity, time.time() - start_time


def evaluate(epoch, model, dataloader, args, mode="val"):
    model.eval()
    losses = []

    total_loss = 0.0
    total_iters = 0

    start_time = time.time()

    with torch.no_grad():
        for idx, batch in enumerate(
            tqdm(dataloader, desc="Evaluation", disable=(not args.progress_bar))
        ):
            batch = to_device(batch, args.device)

            
            hidden_states = model.initial_states(batch["source"].shape[0])
            log_probas, _ = model(batch["source"], hidden_states)
            

            loss = model.loss(log_probas, batch["target"], batch["mask"])
            losses.append(loss.item() * batch["mask"].sum().item())

            total_loss += loss.item()
            total_iters += batch["source"].shape[1]

            if idx % args.print_every == 0:
                tqdm.write(
                    f"[{mode.upper()}] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}"
                )

        mean_loss = np.mean(losses)
        mean_loss /= args.batch_size * dataloader.dataset.max_length

        perplexity = math.exp(mean_loss)

        tqdm.write(
            f"=== [{mode.upper()}] Epoch: {epoch}, Iter: {idx}, Perplexity: {perplexity:.3f} ===>"
        )

    return mean_loss, perplexity, time.time() - start_time


def main(args):
    # Seed the experiment, for repeatability
    seed_experiment(args.seed)

    # Dataloaders
    train_dataset = Wikitext2(args.data_folder, split="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    valid_dataset = Wikitext2(args.data_folder, split="validation")
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_dataset = Wikitext2(args.data_folder, split="test")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Check for the embeddings
    if not os.path.isfile(args.embeddings):
        print("Embeddings not present .. add it to data/ from drive.")

    # Model
    if args.model == "lstm":
        model = LSTM.load_embeddings_from(
            args.embeddings, hidden_size=512, num_layers=args.layers
        )
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
    train_ppls, valid_ppls = [], []
    train_times, valid_times = [], []
    for epoch in range(args.epochs):

        tqdm.write(f"====== Epoch {epoch} ======>")

        loss, ppl, wall_time = train(epoch, model, train_dataloader, optimizer, args)
        train_losses.append(loss)
        train_ppls.append(ppl)
        train_times.append(wall_time)

        loss, ppl, wall_time = evaluate(epoch, model, valid_dataloader, args)
        valid_losses.append(loss)
        valid_ppls.append(ppl)
        valid_times.append(wall_time)

    test_loss, test_ppl, test_time = evaluate(
        epoch, model, test_dataloader, args, mode="test"
    )

    print(f"===== Best validation perplexity: {min(valid_ppls):.3f} =====>")

    return (
        train_losses,
        train_ppls,
        train_times,
        valid_losses,
        valid_ppls,
        valid_times,
        test_loss,
        test_ppl,
        test_time,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run an experiment for assignment 2.")

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--data_folder",
        type=str,
        default="./data",
        help="path to the data folder (default: %(default)s).",
    )
    data.add_argument(
        "--batch_size", type=int, default=2, help="batch size (default: %(default)s)."
    )

    model = parser.add_argument_group("Model")
    model.add_argument(
        "--model",
        type=str,
        choices=["lstm"],
        default="lstm",
        help="name of the model to run (default: %(default)s).",
    )
    model.add_argument(
        "--embeddings",
        type=str,
        default="./data/embeddings.npz",
        help="path to the embeddings file (default: %(default)s).",
    )
    model.add_argument(
        "--layers",
        type=int,
        default=1,
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
        default=1e-3,
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
        default=10,
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

    # Log experiment data
    if args.log is not None:
        save_logs(args, *logs)
