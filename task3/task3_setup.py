import pandas as pd

DATA_PATH = "/course/vista_task3_student_data.csv"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["starthour"] = df["starthour"].astype(int)
    return df

