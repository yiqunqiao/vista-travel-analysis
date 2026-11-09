import json

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from task2_setup import run_setup, build_design_matrix, RANDOM_STATE, MAX_DEPTH, MAX_ITER


def _top_k_records(names, values, k=3):
  
    values = np.array(values)
    indices = np.argsort(-np.abs(values))[:k]

    top_k = []
    for i in indices:
        top_k.append({'feature': names[i], 'value': f'{values[i]:.10f}'})
    return top_k


def part1():
   
    setup_data = run_setup()
    sub = setup_data['sub']
    train_mask = setup_data['train_mask']
    test_mask = setup_data['test_mask']

    X_full = build_design_matrix(sub, train_mask, persinc_col='persinc_mid', occ_encoding='onehot')
    y_full = sub['y']

    X_train, X_test = X_full[train_mask], X_full[test_mask]
    y_train, y_test = y_full[train_mask], y_full[test_mask]

    age_cols = []
    occ_cols = []
    for column_name in X_full.columns:
        if column_name.startswith('age_'):
            age_cols.append(column_name)
        elif column_name.startswith('occ_'):
            occ_cols.append(column_name)

    logreg = LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE).fit(X_train, y_train)
    dt = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=RANDOM_STATE).fit(X_train, y_train)

    result_json = {
        'DecisionTree': {
            'measure': 'feature_importance',
            'top_features': _top_k_records(list(X_full.columns), dt.feature_importances_),
        },
        'LogisticRegression': {
            'measure': 'coefficient',
            'top_features': _top_k_records(list(X_full.columns), logreg.coef_[0]),
        },
    }
    with open('part1_comparisonA.json', 'w') as f:
        json.dump(result_json, f, indent=2)

    result = dict(setup_data)
    result['age_cols'] = age_cols
    result['occ_cols'] = occ_cols
    result['X_full'] = X_full
    result['y_full'] = y_full
    result['logreg'] = logreg
    result['dt'] = dt
    return result

