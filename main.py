import argparse
import time
import sys
import os

from existing_user import DataLoader, run as run_existing
from new_user import run as run_new


def parse_args():
    parser = argparse.ArgumentParser(
        description="Steam Game Recommender System",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--user',
        choices=['new', 'existing'],
        help=(
            "User type:\n"
            "  new       — cold-start flow (no history needed, genre-based)\n"
            "  existing  — hybrid flow (uses your play history)"
        )
    )
    parser.add_argument(
        '--dataset',
        default='steam_dataset_2000users.csv',
        help="Path to the dataset CSV file (default: steam_dataset_2000users.csv)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.dataset):
        print(f"❌ Dataset not found: '{args.dataset}'")
        print("   Place steam_dataset_2000users.csv in the same directory and try again.")
        sys.exit(1)

    # If --user not provided, prompt interactively
    user_type = args.user
    if not user_type:
        print("════════ Steam Game Recommender ════════")
        while True:
            ans = input("Are you a NEW user or an EXISTING user? [new/existing]: ").strip().lower()
            if ans in ('new', 'existing'):
                user_type = ans
                break
            print("Please type 'new' or 'existing'.\n")

    print(f"\nLoading dataset: {args.dataset}")
    loader = DataLoader(args.dataset)
    df = loader.load_and_preprocess()

    start = time.time()

    if user_type == 'new':
        print("\n── New User Mode ──────────────────────────")
        run_new(df)
    else:
        print("\n── Existing User Mode ─────────────────────")
        run_existing(df)

    print(f"\nTotal session time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
