import json
import re

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


DATA_PATH = '/course/person_vista_2024_2025_clean.csv'
RANDOM_STATE = 90049
TEST_SIZE = 0.25
MAX_DEPTH = 4
MAX_ITER = 10000


def _extract_annual_range(category):
    match = re.search(r'\(\$([\d,]+)-\$([\d,]+)\)', category)
    if match is None:
        return None
    lo = int(match.group(1).replace(',', ''))
    hi = int(match.group(2).replace(',', ''))
    return lo, hi


def _household_split(sub, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    unique_hh = sub['hhid'].unique()
    hh_train, hh_test = train_test_split(unique_hh, test_size=test_size, random_state=random_state)
    train_mask = sub['hhid'].isin(hh_train)
    test_mask = sub['hhid'].isin(hh_test)
    return train_mask, test_mask


def one_hot_encode_column(series, prefix, categories=None):
    if categories is None:
        ohe = OneHotEncoder(sparse_output=False)
    else:
        ohe = OneHotEncoder(
            sparse_output=False,
            categories=[categories],
            handle_unknown='ignore'
        )

    encoded = ohe.fit_transform(series.to_frame())
    columns = []
    for category in ohe.categories_[0]:
        columns.append(prefix + '_' + str(category))

    return pd.DataFrame(encoded, columns=columns, index=series.index)


def label_encode_column(series, categories=None):
    le = LabelEncoder()
    if categories is None:
        encoded = le.fit_transform(series)
    else:
        le.fit(categories)

    mapping = {}
    for i, category in enumerate(le.classes_):
        mapping[category] = i

    if categories is not None:
        encoded = series.map(mapping).fillna(-1).astype(int).values

    return encoded, mapping


def load_and_encode_data(path=DATA_PATH):
    df = pd.read_csv(path)
    sub = df[df['anywfh'].isin(['Yes', 'No'])].copy().reset_index(drop=True)
    sub['y'] = (sub['anywfh'] == 'Yes').astype(int)

    persinc_midpoint = {}
    for category in sorted(sub['persinc'].unique()):
        annual_range = _extract_annual_range(category)
        if annual_range is None:
            persinc_midpoint[category] = 0.0
        else:
            lo, hi = annual_range
            persinc_midpoint[category] = (lo + hi) / 2
    sub['persinc_mid'] = sub['persinc'].map(persinc_midpoint)
    return sub, persinc_midpoint


def build_design_matrix(data, train_mask, persinc_col='persinc_mid', occ_encoding='onehot'):
    train_data = data[train_mask]

    age_categories = sorted(train_data['agegroup'].unique())
    sex_categories = sorted(train_data['sex'].unique())

    age_oh = one_hot_encode_column(data['agegroup'], 'age', categories=age_categories)
    sex_oh = one_hot_encode_column(data['sex'], 'sex', categories=sex_categories)

    if occ_encoding == 'onehot':
        occ_categories = sorted(train_data['anzsco1'].unique())
        occ_part = one_hot_encode_column(data['anzsco1'], 'occ', categories=occ_categories)
    elif occ_encoding == 'label_encode':
        occ_categories = sorted(train_data['anzsco1'].unique())
        occ_encoded, _ = label_encode_column(data['anzsco1'], categories=occ_categories)
        occ_part = pd.DataFrame({'occ_label': occ_encoded}, index=data.index)
    else:
        raise ValueError(f"Unknown occ_encoding: {occ_encoding}")

    return pd.concat([age_oh, sex_oh, occ_part, data[[persinc_col]]], axis=1)


def run_setup():
    sub, persinc_midpoint = load_and_encode_data()
    train_mask, test_mask = _household_split(sub)
    train_sub = sub[train_mask]

    features_and_prefixes = [('agegroup', 'age'), ('sex', 'sex'), ('anzsco1', 'occ')]

    feature_info = {}
    for feature, prefix in features_and_prefixes:
        categories = sorted(train_sub[feature].unique())
        encoded_df = one_hot_encode_column(sub[feature], prefix, categories=categories)
        feature_info[feature] = {
            'n_categories': int(len(categories)),
            'one_hot_columns': int(encoded_df.shape[1]),
        }

    occ_categories = sorted(train_sub['anzsco1'].unique())
    occ_label_encoded, occ_label_mapping = label_encode_column(sub['anzsco1'], categories=occ_categories)

    label_encoding_map = {}
    categories_by_code = sorted(occ_label_mapping, key=occ_label_mapping.get)
    for category in categories_by_code:
        label_encoding_map[category] = occ_label_mapping[category]
    feature_info['anzsco1']['label_encoding_map'] = label_encoding_map

    example_row = sub.iloc[0]
    worked_example = {
        'note': 'One row from the data, showing how each of its encoded features actually came out. Category vocabulary is fit on the training rows only.',
    }
    for feature, prefix in features_and_prefixes:
        categories = sorted(train_sub[feature].unique())
        encoded_df = one_hot_encode_column(sub[feature], prefix, categories=categories)
        example_encoded_row = encoded_df.iloc[0]
        active_column = None
        for column_name in encoded_df.columns:
            if example_encoded_row[column_name] == 1:
                active_column = column_name
        worked_example[feature] = {
            'true_category': example_row[feature],
            'one_hot_active_column': active_column,
        }
    worked_example['anzsco1']['label_encoded_value'] = int(occ_label_encoded[0])

    out = feature_info
    out['worked_example'] = worked_example

    with open('task2_setup_encoding_check.json', 'w') as f:
        json.dump(out, f, indent=2)

    return {
        'sub': sub,
        'persinc_midpoint': persinc_midpoint,
        'occ_label_mapping': occ_label_mapping,
        'train_mask': train_mask,
        'test_mask': test_mask,
    }

