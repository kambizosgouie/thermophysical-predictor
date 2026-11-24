import matplotlib.pyplot as plt
import numpy as np


def correlation_figure(data_frame, feature_names, target_names):
    figure, axes = plt.subplots(
        1, 2,
        figsize=(max(10, 2.5 * len(feature_names)), max(2.5, 0.6 * len(target_names) + 1.5)),
    )
    for axis, method in zip(axes, ("pearson", "spearman")):
        correlations = data_frame[feature_names + target_names].corr(method=method)
        values = correlations.loc[target_names, feature_names].values
        image = axis.imshow(values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        axis.set_xticks(range(len(feature_names)), feature_names)
        axis.set_yticks(range(len(target_names)), target_names)
        for row_index, row in enumerate(values):
            for column_index, value in enumerate(row):
                axis.text(
                    column_index, row_index, f"{value:.2f}",
                    ha="center", va="center", fontsize=11,
                    color="white" if abs(value) > 0.5 else "black",
                )
        axis.set_title(f"{method.title()} Correlation")
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    return figure


def parity_figure(x_train, x_test, y_train, y_test, model, target_name):
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, datasets, subtitle in zip(
        axes,
        [
            [(y_test, test_predictions, "Test", "tab:orange")],
            [
                (y_train, train_predictions, "Train", "tab:blue"),
                (y_test, test_predictions, "Test", "tab:orange"),
            ],
        ],
        ("Test data", "Full dataset (train + test)"),
    ):
        for values, predictions, label, color in datasets:
            axis.scatter(
                values, predictions, label=label, color=color,
                alpha=0.7, edgecolor="k", s=30,
            )
        minimum = min(min(np.min(values), np.min(predictions)) for values, predictions, _, _ in datasets)
        maximum = max(max(np.max(values), np.max(predictions)) for values, predictions, _, _ in datasets)
        axis.plot([minimum, maximum], [minimum, maximum], "r--", lw=2)
        axis.set_xlabel(f"True {target_name}")
        axis.set_ylabel(f"Predicted {target_name}")
        axis.set_title(subtitle)
        axis.legend()
        axis.grid(True, alpha=0.3)
    figure.suptitle(f"{target_name} — Parity Plots", fontsize=13)
    figure.tight_layout()
    return figure
