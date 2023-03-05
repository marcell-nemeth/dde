import logging

import numpy as np
from scipy.stats import ks_2samp, describe
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


class UncertaintyKSExperiment(BaselineExperiment):
    def __init__(self, model, dataset, k, significance_thresh, param_grid=None, pca=False):
        super().__init__(model, dataset, param_grid)
        self.name = "Method 2 (Uncertainty-KS)"
        self.k = k
        self.significance_thresh = significance_thresh
        self.ref_distributions = []
        self.det_distributions = []
        self.p_vals = []
        self.pca = pca

    @staticmethod
    def make_kfold_predictions(X, y, model, dataset, k, pca):
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
        # NOTE - need to think through if this should be a pipeline with MinMaxScaler...???

        splitter = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
        # splitter = LeaveOneOut()

        preds = np.array([])
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
                # instantiate training pipeline
                pipe = Pipeline(
                    steps=[
                        ("clf", model),
                    ]
                )

            # fit it
            pipe.fit(X.iloc[train_indicies], y.iloc[train_indicies])

            # score it on this Kfold's test data
            y_preds_split = pipe.predict_proba(X.iloc[test_indicies])
            y_preds_split_posclass_proba = y_preds_split[:, 1]
            preds = np.append(preds, y_preds_split_posclass_proba)

            # get accuracy for split
            split_ACC = pipe.score(X.iloc[test_indicies], y.iloc[test_indicies])
            split_ACCs = np.append(split_ACCs, split_ACC)

        logger.info(f"FINAL SHAPE kfold preds: {preds.shape}")

        return preds, split_ACCs

    def get_reference_response_distribution(self):

        # get data in reference window
        window_idx = self.reference_window_idx
        logger.info(f"GETTING REFERENCE DISTRIBUTION FOR WINDOW: {window_idx}")
        X_train, y_train = self.dataset.get_window_data(window_idx, split_labels=True)

        # perform kfoldsplits to get predictions
        preds, split_ACCs = self.make_kfold_predictions(
            X_train, y_train, self.model, self.dataset, self.k, self.pca
        )

        ref_ACC = np.mean(split_ACCs)
        ref_ACC_SD = np.std(split_ACCs)

        return preds, ref_ACC, ref_ACC_SD

    def get_detection_response_distribution(self):

        # get data in prediction window
        window_idx = self.detection_window_idx
        logger.info(f"GETTING DETECTION DISTRIBUTION FOR WINDOW: {window_idx}")
        X_test, y_test = self.dataset.get_window_data(window_idx, split_labels=True)

        # use trained model to get response distribution
        preds = self.trained_model.predict_proba(X_test)[:, 1]

        # get accuracy for detection window
        det_ACC = self.evaluate_model_aggregate(window="detection")

        return preds, det_ACC

    @staticmethod
    def perform_ks_test(dist1, dist2):
        return ks_2samp(dist1, dist2)

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
        """Response Uncertainty Experiment

        This experiment uses a KS test to detect changes in the target/response distribution between
        the reference and detection windows.

        Logic flow:
            - Train on initial reference window
            - Perform Stratified KFold to obtain prediction distribution on reference window
            - Use trained model to generate predictions on detection window
            - Perform statistical test (KS) between reference and detection window response distributions
                - If different, retrain and update both windows
                - If from same distribution, update detection window and repeat

        """
        logger.info(
            f"-------------------- Started SQSI Model Replacement Run --------------------"
        )
        self.train_model_gscv(window="reference", gscv=True)

        CALC_REF_RESPONSE = True

        for i, split in enumerate(self.dataset.splits):

            if i > self.reference_window_idx:
                logger.info(f"Dataset index of split end: {self.dataset.splits[i]}")

                logger.info(
                    f"Need to calculate Reference response distribution? - {CALC_REF_RESPONSE}"
                )

                # log actual score on detection window
                self.experiment_metrics["scores"].extend(
                    self.evaluate_model_incremental(n=10)
                )

                # get reference window response distribution with kfold + detection response distribution
                if CALC_REF_RESPONSE:
                    (
                        ref_response_dist,
                        ref_ACC,
                        ref_ACC_SD,
                    ) = self.get_reference_response_distribution()
                det_response_dist, det_ACC = self.get_detection_response_distribution()

                logger.info(f"REFERENCE STATS: {describe(ref_response_dist)}")
                logger.info(f"DETECTION STATS: {describe(det_response_dist)}")

                logger.info(f"Dataset Split: {i}")
                logger.info(f"REFERENCE STATS: {describe(ref_response_dist)}")
                logger.info(f"DETECTION STATS: {describe(det_response_dist)}")

                self.ref_distributions.append(ref_response_dist)
                self.det_distributions.append(det_response_dist)

                # compare distributions
                ks_result = self.perform_ks_test(
                    dist1=ref_response_dist, dist2=det_response_dist
                )
                self.p_vals.append(ks_result.pvalue)

                logger.info(f"KS Test: {ks_result}")

                significant_change = (
                    True if ks_result[1] < self.significance_thresh else False
                )
                self.drift_signals.append(significant_change)

                # compare accuracies to see if detection was false alarm
                # i.e. check if change in accuracy is significant
                delta_ACC = np.absolute(det_ACC - ref_ACC)
                threshold_ACC = 3 * ref_ACC_SD  # considering outside 3 SD significant
                significant_ACC_change = True if delta_ACC > threshold_ACC else False
                self.drift_occurences.append(significant_ACC_change)

                if significant_change:
                    # reject null hyp, distributions are NOT identical --> retrain
                    self.train_model_gscv(window="detection", gscv=True)
                    self.update_reference_window()
                    CALC_REF_RESPONSE = True
                    _ks_result_report = "FAILED"
                else:
                    CALC_REF_RESPONSE = False
                    _ks_result_report = "PASSED"

                self.update_detection_window()

                logger.info(f"KS Test Result: {_ks_result_report} | {ks_result}")

        self.calculate_label_expense()
        self.calculate_train_expense()
        self.calculate_errors()
