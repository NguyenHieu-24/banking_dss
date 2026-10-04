import sys
import os
import time
import importlib.util

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")

def load_module(filename):
    """Load một file .py bất kỳ theo đường dẫn, kể cả tên bắt đầu bằng số."""
    filepath = os.path.join(SRC_DIR, filename)
    spec = importlib.util.spec_from_file_location(filename, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

STEPS = [
    ("Step 1 — Categorical Encoding",  "01_categorical_encoding.py"),
    ("Step 2 — Feature Engineering",   "02_feature_engineering.py"),
    ("Step 3 — Feature Selection",     "03_feature_selection.py"),
    ("Step 4 — Data Splitting",        "04_data_splitting.py"),
    ("Step 5 — Model Training",        "05_model_training.py"),
    ("Step 6 — Evaluation & Ensemble", "06_evaluation_and_ensemble.py"),
]


def run_pipeline():
    print("=" * 60)
    print("       LOAN DEFAULT PREDICTION — FULL PIPELINE")
    print("=" * 60)

    total_start = time.time()

    for idx, (label, filename) in enumerate(STEPS, start=1):
        print(f"\n{'=' * 60}")
        print(f"  [{idx}/{len(STEPS)}] {label}")
        print("=" * 60)

        step_start = time.time()
        try:
            module = load_module(filename)
            module.main()
        except Exception as e:
            print(f"\n[Error] {label} failed: {e}")
            print("Pipeline stopped.")
            sys.exit(1)

        elapsed = time.time() - step_start
        print(f"\n✓ Completed {label} ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETED — total time: {total_elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()