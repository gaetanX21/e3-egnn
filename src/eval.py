from models import MPNN, E3EGNN, E3EGNN_edge
from train_eval import evaluate_model
from data import load_qm9, split
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="Script to evaluate a model on the QM9 dataset.")

    parser.add_argument("--model", type=str, default="E3EGNN_edge", help="Model to train. Options: MPNN, E3EGNN, E3EGNN_edge.")
    parser.add_argument("--qm9_dir", type=str, default="datasets/", help="Path to the QM9 dataset.")
    parser.add_argument("--ckpt_path", type=str, help="Path to load the model checkpoint.")

    args = parser.parse_args()

    # load QM9 dataset
    qm9 = load_qm9(args.qm9_dir)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.weights_path))
    print(f'Evaluating model: {model}')
    train_loss = evaluate_model(model, train, device)
    val_loss = evaluate_model(model, val, device)
    test_loss = evaluate_model(model, test, device)
    print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()