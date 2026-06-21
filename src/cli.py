"""Command-line entry point for training and scoring."""
import argparse

from src import train, batch_score


def main():
    p = argparse.ArgumentParser(prog="loan-risk")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train", help="train the risk model")
    s = sub.add_parser("score", help="score a CSV of applicants")
    s.add_argument("input")
    s.add_argument("output", nargs="?", default="reports/batch_decisions.csv")
    args = p.parse_args()

    if args.cmd == "train":
        train.main() if hasattr(train, "main") else None
    elif args.cmd == "score":
        batch_score.main(args.input, args.output)


if __name__ == "__main__":
    main()
