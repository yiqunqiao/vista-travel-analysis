import json

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from task2_setup import build_design_matrix, RANDOM_STATE, MAX_DEPTH, MAX_ITER
from task2_part1 import part1

def _save_part3_json(dt_onehot, dt_label_value, logreg_onehot, logreg_label_value, result_dict):
    out = {
        'feature': 'anzsco1',
        'DecisionTree': {
            'measure': 'feature_importance',
            'onehot': {'value': f'{dt_onehot:.10f}', 'note': 'sum across the 8 one-hot occ_* columns'},
            'label_encoding': {'value': f'{dt_label_value:.10f}', 'note': 'single occ_label column'},
        },
        'LogisticRegression': {
            'measure': 'coefficient',
            'onehot': {'value': f'{logreg_onehot:.10f}', 'note': 'sum of abs(coefficient) across the 8 one-hot occ_* columns'},
            'label_encoding': {'value': f'{logreg_label_value:.10f}', 'note': 'single occ_label column, raw signed coefficient'},
        },
        'returned_keys_check': {
            'logreg_label': 'logreg_label' in result_dict,
            'dt_label': 'dt_label' in result_dict,
            'X_label': 'X_label' in result_dict,
        },
    }
    with open('part3_comparisonC.json', 'w') as f:
        json.dump(out, f, indent=2)
    return result_dict


def part3():
    
    data = part1()

    X_full = data['X_full']
    occ_cols = data['occ_cols']
    logreg = data['logreg']
    dt = data['dt']
    sub = data['sub']
    train_mask = data['train_mask']
    y_full = data['y_full']


    dt_onehot = 0.0
    logreg_onehot = 0.0
    for i, column_name in enumerate(X_full.columns):
        if column_name in occ_cols:
            dt_onehot += float(dt.feature_importances_[i])
            logreg_onehot += abs(float(logreg.coef_[0][i]))

    X_label = build_design_matrix(
        sub, train_mask, persinc_col='persinc_mid',
        occ_encoding='label_encode'
    )
    X_label_train = X_label[train_mask]
    y_train = y_full[train_mask]

    logreg_label = LogisticRegression(
        max_iter=MAX_ITER, random_state=RANDOM_STATE
    ).fit(X_label_train, y_train)
    dt_label = DecisionTreeClassifier(
        max_depth=MAX_DEPTH, random_state=RANDOM_STATE
    ).fit(X_label_train, y_train)

    occ_index = list(X_label.columns).index('occ_label')
    dt_label_value = float(dt_label.feature_importances_[occ_index])
    logreg_label_value = float(logreg_label.coef_[0][occ_index])

    result_dict = data
    result_dict['logreg_label'] = logreg_label
    result_dict['dt_label'] = dt_label
    result_dict['X_label'] = X_label
    
    return _save_part3_json(dt_onehot, dt_label_value, logreg_onehot, logreg_label_value, result_dict)

