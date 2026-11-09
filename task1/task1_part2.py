import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from task1_setup import WFH_DAYS

warnings.filterwarnings("ignore", message=".*keepdims.*", category=FutureWarning)

TIE_BREAK_EPS = 1e-9

def build_config_b_arrays(train_df, test_df):

    train_df = train_df.copy()
    test_df = test_df.copy()

    studying_encoder = LabelEncoder()
    studying_encoder.fit(pd.concat([train_df["studying"], test_df["studying"]]))
    train_studying = studying_encoder.transform(train_df["studying"])
    test_studying = studying_encoder.transform(test_df["studying"])

    industry_encoder = LabelEncoder()
    industry_encoder.fit(pd.concat([train_df["anzsic1"], test_df["anzsic1"]]))
    train_industry = industry_encoder.transform(train_df["anzsic1"])
    test_industry = industry_encoder.transform(test_df["anzsic1"])

    scaler = StandardScaler()
    scaler.fit(train_df[["numstops"]])
    train_num = scaler.transform(train_df[["numstops"]])
    test_num = scaler.transform(test_df[["numstops"]])

    train_wfh = np.zeros((len(train_df), len(WFH_DAYS)))
    test_wfh = np.zeros((len(test_df), len(WFH_DAYS)))
    for i, days in enumerate(train_df["wfh_set"]):
        for j, day in enumerate(WFH_DAYS):
            if day in days:
                train_wfh[i, j] = 1
    for i, days in enumerate(test_df["wfh_set"]):
        for j, day in enumerate(WFH_DAYS):
            if day in days:
                test_wfh[i, j] = 1

    X_train = np.column_stack([
        train_studying, train_industry,
        train_df["carlicence_ord"].to_numpy(), train_num, train_wfh,
        train_df["row_id"].to_numpy(),
    ])
    X_test = np.column_stack([
        test_studying, test_industry,
        test_df["carlicence_ord"].to_numpy(), test_num, test_wfh,
        test_df["row_id"].to_numpy(),
    ])
 
    if "X_train" not in dir():
        X_train = np.zeros((len(train_df), 10))
    if "X_test" not in dir():
        X_test = np.zeros((len(test_df), 10))

    with open("part2_features.json", "w") as f:
        json.dump(
            {
                "n_features": X_train.shape[1],
                "X_train_shape": list(X_train.shape),
                "X_test_shape": list(X_test.shape),
            },
            f,
            indent=2,
        )

    return X_train, X_test


def mixed_distance(a, b):

    nominal_dist = 0
    for i in range(2):
        if a[i] != b[i]:
            nominal_dist += 1

    ordinal_dist = abs(a[2] - b[2])
    numeric_dist = abs(a[3] - b[3])

    intersection = 0
    union = 0
    for i in range(4, 9):
        if a[i] == 1 and b[i] == 1:
            intersection += 1
        if a[i] == 1 or b[i] == 1:
            union += 1

    if union == 0:
        jaccard_dist = 1.0
    else:
        jaccard_dist = 1.0 - intersection / union

    tie_break = TIE_BREAK_EPS * abs(a[9] - b[9])
   
    return float(nominal_dist + ordinal_dist + numeric_dist + jaccard_dist + tie_break)

