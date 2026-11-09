import json
import warnings
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from task1_setup import WFH_DAYS

warnings.filterwarnings("ignore", message=".*keepdims.*", category=FutureWarning)

def build_config_a(train_df, test_df):

    """
    TODO: implement the steps described above.
    """
    catg_cols = ["studying", "anzsic1"]
    catg_encoder = OneHotEncoder(handle_unknown="ignore")
    catg_encoder.fit(train_df[catg_cols])
    train_catg = catg_encoder.transform(train_df[catg_cols]).toarray()
    test_catg = catg_encoder.transform(test_df[catg_cols]).toarray()

    lic_encoder = OneHotEncoder(handle_unknown="ignore")
    lic_encoder.fit(train_df[["carlicence_ord"]])
    train_lic = lic_encoder.transform(train_df[["carlicence_ord"]]).toarray()
    test_lic = lic_encoder.transform(test_df[["carlicence_ord"]]).toarray()

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

    scaler = StandardScaler()
    scaler.fit(train_df[["numstops"]])
    train_num = scaler.transform(train_df[["numstops"]])
    test_num = scaler.transform(test_df[["numstops"]])
    
    """Remove the comment signs (#) from the two lines below when you complete the above steps"""
    X_train = np.hstack([train_catg, train_lic, train_wfh, train_num])
    X_test = np.hstack([test_catg, test_lic, test_wfh, test_num])

    if "X_train" not in dir():
        X_train = np.zeros((len(train_df), 1))
    if "X_test" not in dir():
        X_test = np.zeros((len(test_df), 1))

    with open("part1_features.json", "w") as f:
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

