import logging

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from test_harness.experiments.baseline_experiment import BaselineExperiment

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler("../logs/app.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


class MarginThresholdExperiment(BaselineExperiment):
    def __init__(
        self,
        model,
        dataset,
        k,
        margin_width,
        sensitivity,
        param_grid=None,
        pca=False
    ):
        super().__init__(model, dataset, param_grid)
        self.name = f"Method 4 (Margin-Threshold-S{sensitivity})"
        self.k = k
        self.margin_width = margin_width
        self.sensitivity = sensitivity
        self.drift_signals = []
        self.drift_occurences = []

        self.ref_distributions = []
        self.ref_margins = []
        self.ref_MDs = []
        self.ref_SDs = []
        self.ref_ACCs = []
        self.ref_ACC_SDs = []

        self.det_distributions = []
        self.det_margins = []
        self.det_MDs = []
        self.det_ACCs = []

        self.pca = pca

    @staticmethod
    def make_kfold_predictions(X, y, model, dataset, k, margin_width, pca):
        """A KFold version of LeaveOneOut predictions.

        Rather than performing exhaustive leave-one-out methodology to get predictions
        for each observation, we use a less exhaustive KFold approach.

        When k == len(X), this is equivalent to LeaveOneOut: expensive, but robust. Reducing k
        saves computation, but reduces robustness of model.

        Args:
            X (pd.Dataframe) - features in evaluation window
            y (pd.Series) - labels in evaluation window
            k (int) - number of folds
            type (str) - specified kfold or LeaveOneOut split methodology

        Returns:
            preds (np.array) - an array of predictions for each X in the input (NOT IN ORDER OF INPUT)

        """

        splitter = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)

        preds = np.array([])
        pred_margins = np.array([])
        split_MDs = np.array([])
        split_ACCs = np.array([])

        for train_indicies, test_indicies in splitter.split(X, y):

            if not pca:
                # create column transformer
                column_transformer = ColumnTransformer(
                    [
                        (
                            "continuous",
                            StandardScaler(),
                            dataset.column_mapping["numerical_features"],
                        ),
                        (
                            "categorical",
                            "passthrough",
                            dataset.column_mapping["categorical_features"],
                        ),
                    ]
                )

                # instantiate training pipeline
                pipe = Pipeline(
                    steps=[
                        ("scaler", column_transformer),
                        ("clf", model),
                    ]
                )

            else:
                pipe = Pipeline(
                    steps=[
                        ("clf", model),
                    ]
                )

            # fit it
            pipe.fit(X.iloc[train_indicies], y.iloc[train_indicies])

            # score it on this Kfold's test data
            y_preds_split = pipe.predict_proba(X.iloc[test_indicies])

            # get positive class prediction
            y_preds_split_posclass_proba = y_preds_split[:, 1]
            preds = np.append(preds, y_preds_split_posclass_proba)

            # get pred margins
            # https://github.com/SeldonIO/alibi-detect/blob/86dc3148ee5a3726fb6229d5369c38e7e97b6040/alibi_detect/cd/preprocess.py#L49
            top_2_probs = -np.partition(-y_preds_split, kth=1, axis=-1)
            diffs = top_2_probs[:, 0] - top_2_probs[:, 1]
            pred_margins = np.append(pred_margins, diffs)

            # get MD for split
            split_MD = (
                pd.Series(diffs < margin_width)
                .astype(int)
                .value_counts(normalize=True)[1]
            )
            split_MDs = np.append(split_MDs, split_MD)

            # get accuracy for split
            split_ACC = pipe.score(X.iloc[test_indicies], y.iloc[test_indicies])
            split_ACCs = np.append(split_ACCs, split_ACC)

        return preds, pred_margins, split_MDs, split_ACCs

    def get_reference_response_distribution(self):

        # get data in reference window
        window_idx = self.reference_window_idx
        logger.info(f"GETTING REFERENCE DISTRIBUTION FOR WINDOW: {window_idx}")
        X_train, y_train = self.dataset.get_window_data(window_idx, split_labels=True)

        logger.info(f"SELF MODEL: {self.model}")

        # perform kfoldsplits to get predictions
        preds, pred_margins, split_MDs, split_ACCs = self.make_kfold_predictions(
            X_train, y_train, self.model, self.dataset, self.k, self.margin_width, self.pca
        )

        ref_MD = np.mean(split_MDs)
        ref_SD = np.std(split_MDs)

        ref_ACC = np.mean(split_ACCs)
        ref_ACC_SD = np.std(split_ACCs)

        return preds, pred_margins, ref_MD, ref_SD, ref_ACC, ref_ACC_SD

    def get_detection_response_distribution(self):

        # get data in prediction window
        window_idx = self.detection_window_idx
        logger.info(f"GETTING DETECTION DISTRIBUTION FOR WINDOW: {window_idx}")
        X_test, y_test = self.dataset.get_window_data(window_idx, split_labels=True)

        # use trained model to get response distribution
        y_preds_split = self.trained_model.predict_proba(X_test)
        preds = y_preds_split[:, 1]

        # get pred margins
        # https://github.com/SeldonIO/alibi-detect/blob/86dc3148ee5a3726fb6229d5369c38e7e97b6040/alibi_detect/cd/preprocess.py#L49
        top_2_probs = -np.partition(-y_preds_split, kth=1, axis=-1)
        pred_margins = top_2_probs[:, 0] - top_2_probs[:, 1]

        det_MD = (
            pd.Series(pred_margins < self.margin_width)
            .astype(int)
            .value_counts(normalize=True)[1]
        )

        # get accuracy for detection window
        det_ACC = self.evaluate_model_aggregate(window="detection")

        return preds, pred_margins, det_MD, det_ACC

    def calculate_errors(self):

        self.false_positives = [
            True if self.drift_signals[i] and not self.drift_occurences[i] else False
            for i in range(len(self.drift_signals))
        ]
        self.false_negatives = [
            True if not self.drift_signals[i] and self.drift_occurences[i] else False
            for i in range(len(self.drift_signals))
        ]

    def run(self):
        """Response Margin Threshold Experiment

        This experiment uses a threshold/sensitivity to detect changes in the margin of the target/response distribution between
        the reference and detection windows.

        Logic flow:
            - Train on initial reference window
            - Perform Stratified KFold to obtain "margin" values for each prediction by subtracting class 1 and class 0 confidence scores on reference window
                - Apply a user defined "margin threshold" to classify each observation as "in-margin" or "out-of-margin"
                - Use these population values calculate a MD metric for the given split (MD = # samples in-margin / # samples total)
                - Summarize the kfold split MD metrics into:
                    - The expected margin density (MD_reference): average MD over k-splits
                    - The acceptable deviation of the MD metric (SD_reference): standard deviation of MD over k-splits
            - Use trained model to generate predictions on detection window + calculate MD_detection
            - Compare the MD_reference to MD_detection
                - Check if MD_detection deviates by more than S standard deviations from MD_reference, then a concept drift is detected
                - S is a sensitivity parameter set by user
            - If drift is detected, retrain and update both windows
            - If drift is not-detected, update detection window and repeat

        """
        logger.info(
            f"-------------------- Started Response Margin Threshold Experiment Run --------------------"
        )
        self.train_model_gscv(window="reference", gscv=True)

        CALC_REF_RESPONSE = True

        for i, split in enumerate(self.dataset.splits):

            if i > self.reference_window_idx:
                logger.info(f"Dataset index of split end: {self.dataset.splits[i]}")

                logger.info(f"Need to update MD_reference? - {CALC_REF_RESPONSE}")

                # log actual score on detection window
                self.experiment_metrics["scores"].extend(
                    self.evaluate_model_incremental(n=10)
                )

                # get reference window response distribution with kfold + detection response distribution
                if CALC_REF_RESPONSE:
                    (
                        ref_response_dist,
                        ref_response_margins,
                        ref_MD,
                        ref_SD,
                        ref_ACC,
                        ref_ACC_SD,
                    ) = self.get_reference_response_distribution()

                (
                    det_response_dist,
                    det_response_margins,
                    det_MD,
                    det_ACC,
                ) = self.get_detection_response_distribution()

                # save reference window items
                self.ref_distributions.append(ref_response_dist)
                self.ref_margins.append(ref_response_margins)
                self.ref_MDs.append(ref_MD)
                self.ref_SDs.append(ref_SD)
                self.ref_ACCs.append(ref_ACC)
                self.ref_ACC_SDs.append(ref_ACC_SD)

                # save detection window items
                self.det_distributions.append(det_response_dist)
                self.det_margins.append(det_response_margins)
                self.det_MDs.append(det_MD)
                self.det_ACCs.append(det_ACC)

                # compare margin densities to detect drift
                delta_MD = np.absolute(det_MD - ref_MD)
                threshold = self.sensitivity * ref_SD
                significant_MD_change = True if delta_MD > threshold else False
                self.drift_signals.append(significant_MD_change)

                logger.info(
                    f"Significant Change in Margin Density: {significant_MD_change}"
                )
                logger.info(f"Change in MD: {delta_MD}")
                logger.info(
                    f"Sensitivity: {self.sensitivity} | Ref_SD: {ref_SD} | Threshold: {threshold}"
                )

                logger.info(
                    f"Significant Change in Margin Density: {significant_MD_change}"
                )
                logger.info(f"Change in MD: {delta_MD}")
                logger.info(
                    f"Sensitivity: {self.sensitivity} | Ref_SD: {ref_SD} | Threshold: {threshold}"
                )

                # compare accuracies to see if detection was false alarm
                # i.e. check if change in accuracy is significant
                delta_ACC = np.absolute(det_ACC - ref_ACC)
                threshold_ACC = 3 * ref_ACC_SD  # considering outside 3 SD significant
                significant_ACC_change = True if delta_ACC > threshold_ACC else False
                self.drift_occurences.append(significant_ACC_change)

                if significant_MD_change:
                    self.train_model_gscv(window="detection", gscv=True)
                    self.update_reference_window()
                    CALC_REF_RESPONSE = True
                else:
                    CALC_REF_RESPONSE = False

                self.update_detection_window()

        self.calculate_label_expense()
        self.calculate_train_expense()
        self.calculate_errors()
