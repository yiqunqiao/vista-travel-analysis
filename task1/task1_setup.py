import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PERSON_PATH = "/course/person_vista_2024_2025_clean.csv"

CARLICENCE_ORDER = {
    "No Car Licence": 0,
    "Learners Permit": 1,
    "Red Probationary Licence": 2,
    "Green Probationary Licence": 3,
    "Full Licence": 4,
}

WFH_COLS = ["wfhmon", "wfhtue", "wfhwed", "wfhthu", "wfhfri"]
WFH_DAYS = ["mon", "tue", "wed", "thu", "fri"]

RANDOM_STATE = 90049
TEST_SIZE = 0.25

def build_dataframe():
    
    df = pd.read_csv(PERSON_PATH)
    print(df["wfhmon"].unique())
    df = df[df["wfhmon"] != "Not in Work Force"].copy()
    
    df["carlicence_ord"] = df["carlicence"].map(CARLICENCE_ORDER)
 
    def day_set(row):
        days = set()
        for i in range(len(WFH_COLS)):
            if row[WFH_COLS[i]] == "Yes":
                days.add(WFH_DAYS[i])
        return days
    
    df["wfh_set"] = df.apply(day_set, axis=1)

    if "carlicence_ord" not in df.columns:
        df["carlicence_ord"] = 0
    if "wfh_set" not in df.columns:
        df["wfh_set"] = [frozenset() for _ in range(len(df))]

    df["row_id"] = np.arange(len(df))

    keep_cols = [
        "row_id", "hhid", "persid", "pt_user",
        "studying", "anzsic1", "carlicence_ord", "wfh_set", "numstops",
    ]
    df = df[keep_cols]

    df_to_save = df.copy()
    df_to_save["wfh_set"] = df_to_save["wfh_set"].apply(lambda s: ",".join(sorted(s)))
    df_to_save.to_csv("setup_dataframe.csv", index=False)

    return df


def household_split(df):

    unique_hh = df["hhid"].unique()
    train_hh, test_hh = train_test_split(unique_hh, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_df = df[df["hhid"].isin(train_hh)].reset_index(drop=True)
    test_df = df[df["hhid"].isin(test_hh)].reset_index(drop=True)
    
    return train_df, test_df

