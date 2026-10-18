"""Run the full feature-extraction pipeline (Tasks 0-5) for both segmentations (4s windows, 60s trials)."""
import json

import task0_metadata
import task1_de
import task2_hjorth
import task3_asymmetry
import task4_mse
import task5_microstates
from bcc.features import SANITY_DIR, SEGMENTATIONS

# Dependency order: task3 reads task1's saved CSVs. MSE and microstates are cached per subject.
TASKS = [task1_de, task2_hjorth, task3_asymmetry, task4_mse, task5_microstates]


def main():
    print(f"\n== {task0_metadata.__doc__}")
    summaries = {task0_metadata.__name__: task0_metadata.run()[1]}
    for seg in SEGMENTATIONS:
        summaries[seg] = {}
        for module in TASKS:
            print(f"\n== [{seg}] {module.__doc__}")
            summaries[seg][module.__name__] = module.run(seg)[1]
    (SANITY_DIR / "run_summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    print("\nDone. To recompute MSE or microstates, delete their folder under outputs/features/ or "
          "outputs/features_60s/ (and outputs/models/ for the group maps).")


if __name__ == "__main__":
    main()
