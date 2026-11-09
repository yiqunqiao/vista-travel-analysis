import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from task2_part3 import part3


def _save_confusion_matrices_png(results):
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes_list = axes.flatten() 
    for i in range(4):
        model_name, encoding, y_true, y_pred = results[i]
        ax = axes_list[i]
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=['No', 'Yes'], cmap='Blues',
            ax=ax, colorbar=False,
        )
        ax.set_title(f'{model_name} ({encoding})')

    fig.tight_layout()
    fig.savefig('task2_evaluation_confusion_matrices.png')
    plt.close(fig)


def _save_evaluation_json(raw_metrics):
    formatted = {}
    for model_name in raw_metrics:
        formatted[model_name] = {}
        for encoding in raw_metrics[model_name]:
            metrics = raw_metrics[model_name][encoding]
            formatted[model_name][encoding] = {
                'accuracy': f'{metrics["accuracy"]:.10f}',
                'precision_macro': f'{metrics["precision_macro"]:.10f}',
                'recall_macro': f'{metrics["recall_macro"]:.10f}',
                'f1_macro': f'{metrics["f1_macro"]:.10f}',
            }
    with open('task2_evaluation_results.json', 'w') as f:
        json.dump(formatted, f, indent=2)
    return formatted


def evaluation():
    data = part3()

    logreg = data['logreg']
    dt = data['dt']
    logreg_label = data['logreg_label']
    dt_label = data['dt_label']
    X_full = data['X_full']
    X_label = data['X_label']
    test_mask = data['test_mask']
    y_full = data['y_full']


    X_test = X_full[test_mask]
    X_label_test = X_label[test_mask]
    y_test = y_full[test_mask]

    combos = [
        ('LogisticRegression', 'onehot', logreg, X_test),
        ('LogisticRegression', 'label_encoding', logreg_label, X_label_test),
        ('DecisionTree', 'onehot', dt, X_test),
        ('DecisionTree', 'label_encoding', dt_label, X_label_test),
    ]

    confusion_inputs = []
    raw_metrics = {}
    for model_name, encoding, model, test_data in combos:
        y_pred = model.predict(test_data)
        confusion_inputs.append((model_name, encoding, y_test, y_pred))

        if model_name not in raw_metrics:
            raw_metrics[model_name] = {}
        raw_metrics[model_name][encoding] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision_macro': float(precision_score(
                y_test, y_pred, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(
                y_test, y_pred, average='macro', zero_division=0)),
            'f1_macro': float(f1_score(
                y_test, y_pred, average='macro', zero_division=0)),
        }
    
    _save_confusion_matrices_png(confusion_inputs)
    return _save_evaluation_json(raw_metrics)

