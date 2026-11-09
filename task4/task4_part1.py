import json

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from task4_setup import run_setup


def _save_part1_json(mse_train, mse_test):
  
    out = {
        'model': 'LinearRegression (cumdist only)',
        'trained_on': 'Dataset1',
        'mse_on_Dataset1_train': f'{mse_train:.10f}',
        'mse_on_Dataset3_test': f'{mse_test:.10f}',
    }
    with open('part1_comparisonA.json', 'w') as f:
        json.dump(out, f, indent=2)
    return out


def part1():
  
    data = run_setup()
    dataset1 = data['dataset1']
    dataset3 = data['dataset3']

    X_train = dataset1[['cumdist_num']].values
    y_train = dataset1['travtime'].values
    X_test = dataset3[['cumdist_num']].values
    y_test = dataset3['travtime'].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    mse_train = mean_squared_error(y_train, model.predict(X_train))
    mse_test = mean_squared_error(y_test, model.predict(X_test))
    _save_part1_json(mse_train, mse_test)

    data['model_single_dataset1'] = model
    return data

