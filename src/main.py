from models import MPNN, E3EGNN, E3EGNN_edge, LinReg
from train_eval import run_experiment
from data import load_qm9, split
import argparse
import torch
import wandb


def main():
    parser = argparse.ArgumentParser(description="Script to train a model on the QM9 dataset.")

    parser.add_argument("--model", type=str, default="E3EGNN_edge", help="Model to train. Options: MPNN, E3EGNN, E3EGNN_edge.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--use_wandb", action='store_true', help="Use Weights & Biases for logging.")
    parser.add_argument("--qm9_dir", type=str, default="datasets/", help="Path to the QM9 dataset.")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt/", help="Path to save the model weights.")
    parser.add_argument("--use_scheduler", action='store_true', help="Use LROnPlateau scheduler.")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to load model checkpoint if retraining.")
    parser.add_argument("--warm_optimizer", action='store_true', help="Loads the optimizer state from the checkpoint.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")

    args = parser.parse_args()

    # load QM9 dataset
    qm9 = load_qm9(args.qm9_dir)
    train, val, test = split(qm9, train_ratio=0.8, batch_size=args.batch_size)

    # initialize model (with default parameters)
    model = None
    if args.model == "MPNN":
        model = MPNN()
    elif args.model == "E3EGNN":
        model = E3EGNN()
    elif args.model == "E3EGNN_edge":
        model = E3EGNN_edge()
    elif args.model == "LinReg":
        model = LinReg()
    else:
        raise ValueError(f"Invalid model argument: {args.model}")

    starting_epoch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # load model checkpoint
    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])
        starting_epoch = ckpt['epoch']
        if args.warm_optimizer:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f"Model checkpoint loaded from {args.ckpt_path}")

    # train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_experiment(model, train, val, device, starting_epoch, args.epochs, optimizer, args.use_wandb, args.use_scheduler)
    # save model weights
    name = wandb.run.name
    ckpt_path = args.ckpt_dir + name + '.pt'
    torch.save({
        'epoch': args.epochs + starting_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, ckpt_path)
    print(f"Model weights saved to {ckpt_path}")


if __name__ == "__main__":
    main()