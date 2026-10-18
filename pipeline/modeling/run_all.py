"""Run the Week 4 modeling stage (Tasks 1-2) on the 60s feature set."""
import task1_train
import task2_evaluate
from bcc.analysis import load_all


def main():
    print(f"\n== {task1_train.__doc__}")
    task1_train.run(*load_all("60s"))
    print(f"\n== {task2_evaluate.__doc__}")
    task2_evaluate.run()


if __name__ == "__main__":
    main()
