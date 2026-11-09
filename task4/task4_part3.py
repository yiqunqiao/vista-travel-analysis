import json

from sklearn.metrics import mean_squared_error

from task4_part2 import part2


def _save_part3_json(mse_dataset3, mse_dataset4, centers):
    out = {
        'model': 'LinearRegression (cumdist and starthour), trained on Dataset2',
        'mse_on_Dataset3_test_same_distribution': f'{mse_dataset3:.10f}',
        'mse_on_Dataset4_test_shifted_distribution': f'{mse_dataset4:.10f}',
        'feature_centers': {
            'note': 'Mean of each feature -- compare Dataset4 to Dataset2/Dataset3 to see how far the shifted data sits from what the model was trained on.',
            'Dataset2_mean_cumdist_km': centers['Dataset2_cumdist'],
            'Dataset3_mean_cumdist_km': centers['Dataset3_cumdist'],
            'Dataset4_mean_cumdist_km': centers['Dataset4_cumdist'],
            'Dataset2_mean_starthour': centers['Dataset2_starthour'],
            'Dataset3_mean_starthour': centers['Dataset3_starthour'],
            'Dataset4_mean_starthour': centers['Dataset4_starthour'],
        },
    }
    with open('part3_comparisonC.json', 'w') as f:
        json.dump(out, f, indent=2)
    return out


def part3():
    data = part2()
    model = data['model_both_dataset2']
    dataset2 = data['dataset2']
    dataset3 = data['dataset3']
    dataset4 = data['dataset4']

    X_dataset3 = dataset3[['cumdist_num', 'starthour']].values
    X_dataset4 = dataset4[['cumdist_num', 'starthour']].values
    mse_dataset3 = mean_squared_error(
        dataset3['travtime'].values, model.predict(X_dataset3)
    )
    mse_dataset4 = mean_squared_error(
        dataset4['travtime'].values, model.predict(X_dataset4)
    )

    centers = {
        'Dataset2_cumdist': f"{dataset2['cumdist_num'].mean():.10f}",
        'Dataset3_cumdist': f"{dataset3['cumdist_num'].mean():.10f}",
        'Dataset4_cumdist': f"{dataset4['cumdist_num'].mean():.10f}",
        'Dataset2_starthour': f"{dataset2['starthour'].mean():.10f}",
        'Dataset3_starthour': f"{dataset3['starthour'].mean():.10f}",
        'Dataset4_starthour': f"{dataset4['starthour'].mean():.10f}",
    }
    _save_part3_json(mse_dataset3, mse_dataset4, centers)
    return {}

