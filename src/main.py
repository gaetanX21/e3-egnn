from models import MPNN, E3EGNN, E3EGNN_edge
from train_eval import run_experiment
from data import load_qm9, split
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="Script to train a model on the QM9 dataset.")

    parser.add_argument("--model", type=str, default="E3EGNN_edge", help="Model to train. Options: MPNN, E3EGNN, E3EGNN_edge.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--use_wandb", type=bool, default=True, help="Whether to use Weights & Biases for logging.")
    parser.add_argument("--qm9_path", type=str, help="Path to the QM9 dataset.")
    parser.add_argument("--weights_path", type=str, help="Path to save the model weights.")

    args = parser.parse_args()

    # load QM9 dataset
    qm9 = load_qm9(args.qm9_path)
    train, val, test = split(qm9, train_ratio=0.8)

    # initialize model (with default parameters)
    model = None
    if args.model == "MPNN":
        model = MPNN()
    elif args.model == "E3EGNN":
        model = E3EGNN()
    elif args.model == "E3EGNN_edge":
        model = E3EGNN_edge()
    else:
        raise ValueError(f"Invalid model argument: {args.model}")

    # train model
    run_experiment(model, train, val, test, args.epochs, args.lr, args.use_wandb)
    # save model weights
    torch.save(model.state_dict(), args.weights_path)
    print(f"Model weights saved to {args.weights_path}")


if __name__ == "__main__":
    main()