import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from labellines import labelLines


def format_experimental_scores(experiment):
    return (
        pd.DataFrame(experiment.experiment_metrics["scores"])
        .set_index(0)
        .rename(columns={1: f"{experiment.name} PCA={experiment.dataset.pca_dim}"})
    )


def plot_experiment_error(experiment, retrain_points=True):
    """Utility for plotting single experiment's cumulative error"""

    scores_df = format_experimental_scores(experiment)

    ax = scores_df.plot(
        figsize=(15, 7),
        title=f"Cumulative Score over Experiment \n \
            Label Expense: {experiment.experiment_metrics['label_expense']['num_labels_requested']} \n \
            Total Train Time: {experiment.experiment_metrics['total_train_time']}",
    )

    retrainings = [
        experiment.dataset.get_split_idx(window_idx)
        for window_idx in [
            train_record["window_idx"]
            for train_record in experiment.experiment_metrics["training"]
        ][1:]
    ]

    if retrain_points:
        [
            ax.axvline(i, color="black", linestyle=":", linewidth=0.75)
            for i in retrainings
            if i != 0
        ]
    labelLines(plt.gca().get_lines(), align=True)
    plt.show()


def plot_multiple_experiments(experiments, change_points=None):
    """Utility for plotting multiple experiment's cumulative error"""

    exp_dfs = [format_experimental_scores(experiment) for experiment in experiments]

    ax = pd.concat(exp_dfs, axis=1).plot(
        figsize=(15, 7),
        title="Cumulative Accuracy on Data Stream by Drift Detection Method",
        xlabel="Observations",
        ylabel="Cumulative Accuracy",
    )

    if change_points:
        [
            ax.axvline(i, color="black", linestyle=":", linewidth=0.75)
            for i in change_points
            if i != 0
        ]

    plt.legend(bbox_to_anchor=(1, 0.5))
    labelLines(plt.gca().get_lines(), align=True)
    plt.show()
    # plt.savefig('../figures/multiple_experiments.png')


def aggregate_experiment_metrics(experiments):
    """Utility for aggregating metrics given multiple experiments"""

    metrics = []
    for exp in experiments:

        metrics.append(
            {
                "experiment": exp.name,
                "drifts_signaled": sum(exp.drift_signals),
                "actual_drift_occurences": sum(exp.drift_occurences),
                "false_positives": sum(exp.false_positives),
                "false_negatives": sum(exp.false_negatives),
                "percent_total_labels": exp.experiment_metrics["label_expense"][
                    "percent_total_labels"
                ],
                "cumulative_accuracy": exp.experiment_metrics["scores"][-1][1],
                "total_train_time": exp.experiment_metrics["total_train_time"],
            }
        )

    return pd.DataFrame(metrics).set_index("experiment")


def plot_response_distributions_bysplit(sqsi_exp):
    """Utility for UncertaintyKSExperiment"""

    df = pd.DataFrame()

    for i in range(len(sqsi_exp.ref_distributions)):
        dists = pd.DataFrame(
            np.stack([sqsi_exp.ref_distributions[i], sqsi_exp.det_distributions[i]]).T,
            columns=["Reference", "Detection"],
        )
        dists["Split"] = i

        df = df.append(dists)

    df_melt = df.melt(id_vars=["Split"], var_name="Window Type")

    g = sns.FacetGrid(df_melt, col="Split", hue="Window Type", col_wrap=4)
    g.map_dataframe(sns.kdeplot, "value", fill=True)
    g.add_legend()


def plot_margin_distributions_bysplit(margin_exp):
    """Utility for UncertaintyX2Experiment"""

    for i in range(len(margin_exp.ref_margins)):

        ref_margin_ex = margin_exp.ref_margins[i]
        det_margin_ex = margin_exp.det_margins[i]

        # plot margin distributions
        ax = pd.DataFrame(
            np.vstack((ref_margin_ex, det_margin_ex)).T,
            columns=["Reference", "Detection"],
        ).plot(kind="kde")

        ax.axvline(
            margin_exp.margin_width, color="black", linestyle=":", linewidth=0.75
        )

        ref_uncertainties = (ref_margin_ex < margin_exp.margin_width).astype(int)
        det_uncertainties = (det_margin_ex < margin_exp.margin_width).astype(int)

        expected = pd.Series(ref_uncertainties).value_counts(normalize=False).tolist()
        observed = pd.Series(det_uncertainties).value_counts(normalize=False).tolist()

        expected_norm = (
            pd.Series(ref_uncertainties).value_counts(normalize=True).tolist()
        )
        observed_norm = (
            pd.Series(det_uncertainties).value_counts(normalize=True).tolist()
        )

        pct_change_in_margin = round(
            (np.absolute(expected_norm[1] - observed_norm[1]) / expected_norm[1]), 4
        )

        x2_test = scipy.stats.chisquare(f_obs=observed, f_exp=expected)

        same_dist = True if x2_test[1] > 0.001 else False
        print(f"Same Distribution: {same_dist}")
        print(f"Expected Distribution: {expected_norm}")
        print(f"Observed Distribution: {observed_norm}")
        print(f"Percent change in margin: {pct_change_in_margin}")
        print(
            f"Number in Margin: Before {expected[1]} | After {observed[1]} | Difference {expected[1]-observed[1]}"
        )
        print(f"Chi-Square Results: {x2_test}")
        print()


def calculate_split_window_distances(experiment, distance_func):

    splits = []

    for i in range(len(experiment.ref_distributions)):

        ref_dist = experiment.ref_distributions[i]
        det_dist = experiment.det_distributions[i]

        distance = distance_func(ref_dist, det_dist)

        splits.append((i, distance))

    return pd.DataFrame(splits, columns=["Split", "Distance"])

