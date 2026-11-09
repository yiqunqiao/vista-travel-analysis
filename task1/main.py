"""
Task 1 — main.py

Orchestrates Setup, part 1, and part 2 by importing their functions --
it does not redefine any logic that already lives in those files.

Run the whole pipeline:
    python main.py
    python main.py all

Run a single part from the command line (each of these runs Setup
first if needed, since part 1 and part 2 both need the train/test
split that Setup produces -- it is not re-implemented, just called
as a dependency):
    python main.py setup
    python main.py part1
    python main.py part2

Or, from a notebook / interactive session:
    import main
    df = main.run_setup()
    train_df, test_df = task1_setup.household_split(df)
    main.run_part1(train_df, test_df)
    main.run_part2(train_df, test_df)
"""

import json
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

import task1_setup
import task1_part1
import task1_part2

# Use the same k for Config A and Config B to make the comparison fair.
# An odd k is used to avoid ties between the two classes (0 and 1).
K = 7


def run_setup():
    """Build the feature dataframe (Setup). Returns the full dataframe
    (not yet split into train/test)."""
    df = task1_setup.build_dataframe()
    print(f"\nSetup done: {len(df)} rows. Saved setup_dataframe.csv")
    return df


def run_part1(train_df, test_df):
    """Build Config A's features and evaluate it with KNN (part 1).
    Returns the macro-F1 score."""
    y_train = train_df["pt_user"].to_numpy()
    y_test = test_df["pt_user"].to_numpy()

    X_train, X_test = task1_part1.build_config_a(train_df.copy(), test_df.copy())
    knn_a = KNeighborsClassifier(n_neighbors=K, metric="euclidean")
    knn_a.fit(X_train, y_train)
    pred_a = knn_a.predict(X_test)
    f1_a = f1_score(y_test, pred_a, average="macro")

    print(f"\npart 1 done: Config A macro-F1 = {round(float(f1_a), 4)}")
    return f1_a


def run_part2(train_df, test_df):
    """Build Config B's features and evaluate it with KNN (part 2).
    Returns the macro-F1 score."""
    y_train = train_df["pt_user"].to_numpy()
    y_test = test_df["pt_user"].to_numpy()

    X_train, X_test = task1_part2.build_config_b_arrays(train_df.copy(), test_df.copy())
    # Use algorithm='brute' because mixed_distance() is a custom distance
    # function designed to compare actual rows from the dataset.
    # Other algorithms may call it on internally generated points that do not
    # have the same format as real rows, which can give incorrect results.
    # Brute force only compares real training/test rows, so it is safe here.
    knn_b = KNeighborsClassifier(n_neighbors=K, metric=task1_part2.mixed_distance, algorithm="brute")
    knn_b.fit(X_train, y_train)

    # mixed_distance() is a custom Python function, so it is slower than
    # built-in metrics like "euclidean". It needs to calculate the distance
    # separately for each train/test pair.
    # Therefore, Part 2 may take several seconds or up to a minute to run.
    # This is normal and does not mean the code is stuck.
    print("\n================================================="
          "\nRunning Config B's KNN (custom distance function)"
          "\nplease wait..."
          "\n=================================================")
    pred_b = knn_b.predict(X_test)
    f1_b = f1_score(y_test, pred_b, average="macro")

    print(f"\npart 2 done: Config B macro-F1 = {round(float(f1_b), 4)}")
    return f1_b


def run_all():
    """Run Setup, then part 1 and part 2 on the same train/test split,
    and save a combined results file."""
    df = run_setup()
    train_df, test_df = task1_setup.household_split(df)

    f1_a = run_part1(train_df, test_df)
    f1_b = run_part2(train_df, test_df)

    results = {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "config_A_macro_f1": round(float(f1_a), 4),
        "config_B_macro_f1": round(float(f1_b), 4),
    }

    with open("task1_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    if arg in ("all",):
        run_all()
    elif arg == "setup":
        run_setup()
    elif arg in ("part1", "part2"):
        # part 1 and part 2 both need a train/test split, which only
        # Setup produces -- this calls run_setup() as a dependency,
        # it doesn't redo Setup's logic itself.
        df = run_setup()
        train_df, test_df = task1_setup.household_split(df)
        if arg == "part1":
            run_part1(train_df, test_df)
        else:
            run_part2(train_df, test_df)
    else:
        raise ValueError(
            f"Unknown argument: {arg!r}. Expected one of: setup, part1, part2, all"
        )
