import json

from task2_part1 import part1

def _save_part2_json(dt_total, logreg_total):
   
    out = [
        {'model': 'DecisionTree', 'total_age_importance': f'{dt_total:.10f}'},
        {'model': 'LogisticRegression', 'total_age_importance': f'{logreg_total:.10f}'},
    ]
    with open('part2_comparisonB.json', 'w') as f:
        json.dump(out, f, indent=2)
    return out


def part2():
    data = part1()

    X_full = data['X_full']
    age_cols = data['age_cols']
    logreg = data['logreg']
    dt = data['dt']


    dt_total = 0.0
    logreg_total = 0.0

    for i, column_name in enumerate(X_full.columns):
        if column_name in age_cols:
            dt_total += float(dt.feature_importances_[i])
            logreg_total += abs(float(logreg.coef_[0][i]))
    
    return _save_part2_json(dt_total, logreg_total)

