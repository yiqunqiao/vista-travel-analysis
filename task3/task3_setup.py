import os
from pathlib import Path
import pandas as pd

DATA_PATH = str(Path(os.environ.get("VISTA_DATA_DIR", Path(__file__).resolve().parents[1] / "data")) / "vista_task3_student_data.csv.gz")


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["starthour"] = df["starthour"].astype(int)
    return df

