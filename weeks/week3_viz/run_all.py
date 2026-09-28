"""Run the Week 3 visualization + correlation check (Tasks 1-3)."""
import task1_class_plots
import task2_embeddings
import task3_correlation
from viz_common import load_all


def main():
    df, cols = load_all()
    for module in [task1_class_plots, task2_embeddings, task3_correlation]:
        print(f"\n== {module.__doc__}")
        module.run(df, cols)


if __name__ == "__main__":
    main()
