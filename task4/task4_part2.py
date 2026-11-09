import json

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from task4_part1 import part1


def _save_part2_json(mse_single_train, mse_single_test, mse_both_train, mse_both_test,
                      gap_dataset1, gap_dataset2):
    out = {
        'trained_on': 'Dataset2',
        'single_feature_cumdist_only': {
            'mse_on_Dataset2_train': f'{mse_single_train:.10f}',
            'mse_on_Dataset3_test': f'{mse_single_test:.10f}',
        },
        'both_features_cumdist_and_starthour': {
            'mse_on_Dataset2_train': f'{mse_both_train:.10f}',
            'mse_on_Dataset3_test': f'{mse_both_test:.10f}',
        },
        'train_test_gap_comparison': {
            'note': 'gap = MSE on Dataset3 (test) minus MSE on the model\'s own training data, using the single-feature model in both cases -- a smaller gap on the larger dataset shows a training-set-size effect.',
            'Dataset1_gap_n_households_30': f'{gap_dataset1:.10f}',
            'Dataset2_gap_n_households_300': f'{gap_dataset2:.10f}',
        },
    }
    with open('part2_comparisonB.json', 'w') as f:
        json.dump(out, f, indent=2)
    return out


def part2():
    data = part1()
    dataset1 = data['dataset1']
    dataset2 = data['dataset2']
    dataset3 = data['dataset3']

    X_single_train = dataset2[['cumdist_num']].values
    X_single_test = dataset3[['cumdist_num']].values
    X_both_train = dataset2[['cumdist_num', 'starthour']].values
    X_both_test = dataset3[['cumdist_num', 'starthour']].values
    y_train = dataset2['travtime'].values
    y_test = dataset3['travtime'].values

    model_single = LinearRegression()
    model_single.fit(X_single_train, y_train)
    model_both = LinearRegression()
    model_both.fit(X_both_train, y_train)

    mse_single_train = mean_squared_error(
        y_train, model_single.predict(X_single_train)
    )
    mse_single_test = mean_squared_error(
        y_test, model_single.predict(X_single_test)
    )
    mse_both_train = mean_squared_error(
        y_train, model_both.predict(X_both_train)
    )
    mse_both_test = mean_squared_error(
        y_test, model_both.predict(X_both_test)
    )

    model_dataset1 = data['model_single_dataset1']
    mse_dataset1_train = mean_squared_error(
        dataset1['travtime'].values,
        model_dataset1.predict(dataset1[['cumdist_num']].values)
    )
    mse_dataset1_test = mean_squared_error(
        y_test, model_dataset1.predict(X_single_test)
    )
    gap_dataset1 = mse_dataset1_test - mse_dataset1_train
    gap_dataset2 = mse_single_test - mse_single_train

    _save_part2_json(
        mse_single_train, mse_single_test, mse_both_train, mse_both_test,
        gap_dataset1, gap_dataset2
    )

    data['model_single_dataset2'] = model_single
    data['model_both_dataset2'] = model_both
    return data

