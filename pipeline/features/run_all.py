"""Run the full Week 2 feature-extraction pipeline (Tasks 0-5)."""
import json

import task0_metadata
import task1_de
import task2_hjorth
import task3_asymmetry
import task4_mse
import task5_microstates
from bcc.features import SANITY_DIR

# Dependency order: task3 reads task1's saved CSVs. MSE and microstates are cached per subject.
TASKS = [task0_metadata, task1_de, task2_hjorth, task3_asymmetry, task4_mse, task5_microstates]


def main():
    summaries = {}
    for module in TASKS:
        print(f"\n== {module.__doc__}")
        summaries[module.__name__] = module.run()[1]
    (SANITY_DIR / "run_summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    print("\nDone. Delete outputs/features/mse/, outputs/features/microstates/ or outputs/models/ to recompute those.")


if __name__ == "__main__":
    main()
