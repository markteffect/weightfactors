from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from weightfactors.utils.exceptions import (ExtremeWeightsError,
                                            WeightsConvergenceError,
                                            extreme_weights)


class GeneralizedRaker:
    """Generalized Rake Weighting algorithm

    Args:
        population_targets: Dict[str, Dict[Any, float]]
            A dictionary mapping column names to a dictionary containing each
                category and its corresponding population targets
                    Example: {"Gender": {"Male": 0.5, "Female": 0.5}}
        raise_on_extreme: bool, optional
            Whether to raise an error when the weight factors are extreme
                according to `cutoffs`, else we raise a warning. Default is False.
        cutoffs: Dict[str, float], optional
            When weights are considered to be extreme. 'lo' is the lower bound (defaults to 0.25)
                and 'hi' is the upper bound (defaults to 4). If `raise_on_extreme` we raise an
                    error if any weight exceeds the cutoffs, otherwise we clip the extremes to the cutoffs
        exclusion_column: str, optional
            a column of 0's and 1's in the input data that denotes which rows should get a weight factor (1)
                and which rows should not (0). Rows that should not get weighted get a weight factor of exactly 1.

    Example:
        >>> dataset = pd.DataFrame({"Gender": [0, 0, 0, 0, 0, 1, 1, 1, 1]})
        >>> targets = {"Gender": {0: 0.5, 1: 0.5}}
        >>> raker = GeneralizedRaker(targets)
        >>> weights = raker.rake(dataset)
        ... [0.9   0.9   0.9   0.9   0.9   1.125 1.125 1.125 1.125]

    References:
        This implementation has been largely inspired by Jérôme Parent-Lévesque's article on the subject:
            https://dev.to/potloc/generalized-raking-for-survey-weighting-2d1d
                https://www.jstor.org/stable/2290793

    Note:
        For readability, the variables in the equation and implementation of the algorithm
        have been renamed in this module as follows:

            - X = `design_matrix`
            - T = `target_vector`
            - H = `identity_matrix`
            - λ = `lagrange_multipliers`
            - w = `weights`

    """

    def __init__(
        self,
        population_targets: Dict[str, Dict[Any, float]],
        raise_on_extreme: bool = False,
        cutoffs: Dict[str, float] = {"lo": 0.25, "hi": 4},
        exclusion_column: Optional[str] = None,
    ):
        self.population_targets = population_targets
        self.raise_on_extreme = raise_on_extreme
        self.cutoffs = cutoffs
        self.exclusion_column = exclusion_column
        self.column_names = list(population_targets.keys())
        self.loss: List[float] = [np.inf]
        self.success: bool = False
        self.weights = np.empty(0)

    def create_design_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Create the design matrix

        Example:
            >>> data = pd.DataFrame({"Gender": [0, 0, 0, 0, 1, 1, 1]})
            >>> raker = GeneralizedRaker({"Gender": {0: 0.5, 1: 0.5}})
            >>> raker.create_design_matrix(data)
            ... [[1 0]
            ...  [1 0]
            ...  [1 0]
            ...  [1 0]
            ...  [0 1]
            ...  [0 1]
            ...  [0 1]]
        """
        dummies = pd.get_dummies(data, columns=self.column_names)
        design_matrix = dummies[dummies.columns.difference(data.columns, sort=False)]
        return np.asarray(design_matrix.values)

    def create_target_vector(self, data: pd.DataFrame) -> np.ndarray:
        """Create the target vector

        Example:
            >>> data = pd.DataFrame({"Gender": [0, 0, 0, 0, 1, 1, 1]})
            >>> raker = GeneralizedRaker({"Gender": {0: 0.5, 1: 0.5}})
            >>> raker.create_target_vector(data)
            ... [3.5 3.5]

        """
        target_vector = []
        for question in self.column_names:
            counts = data[question].value_counts(sort=False)  # Don't sort by frequency
            total_observed_n = counts.sum()
            sorted_counts = dict(sorted(counts.items()))  # Sort keys alphabetically
            for value, _ in sorted_counts.items():
                target_n = total_observed_n * self.population_targets[question][value]
                target_vector.append(target_n)
        return np.asarray(target_vector)

    def validate_exclusion_column(self, data: pd.DataFrame) -> None:
        # Make sure the inclusion column is valid
        if self.exclusion_column:
            if self.exclusion_column not in data.columns:
                raise KeyError(
                    f"There is no column {self.exclusion_column} in the provided dataset"
                )
            unique_values = set(data[self.exclusion_column])
            if not len(unique_values) == 2 or not (
                0 in unique_values and 1 in unique_values
            ):
                raise ValueError(
                    f"Exclusion column '{self.exclusion_column}' is invalid. "
                    "The exclusion column should be a column of 0's and 1's."
                )

    def validate_input(self, data: pd.DataFrame) -> None:
        for key, value in self.population_targets.items():
            # Make sure the values per group add to one
            if not np.isclose(sum([v for _, v in value.items()]), 1.0, atol=1e-4):
                raise ValueError(
                    f"The sum of the population distribution for '{key}' is not 1"
                )
            # Make sure all keys are present in the dataset
            if key not in data.columns:
                raise KeyError(f"There is no column {key} in the provided dataset")
            # Make sure there are no missing values in the columns used for calculating weights
            if data[key].isna().any(axis=None):
                raise ValueError(f"Column {key} contains missing values")
            # Make sure all unique values in the target columns have been mapped
            # It is impossible to set values with observations to a weight of 0
            for k in list(data[key].unique()):
                if k not in list(value.keys()):
                    raise KeyError(
                        f"There are observations for a value in '{key}' that has not been mapped to a population target"
                    )
            # Make sure we have at least 1 observation for each category
            # It is impossible to set values without observations to a weight larger than 0
            for k, v in value.items():
                if k not in data[key].unique() and v > 0:
                    raise KeyError(
                        f"There are no observations for {k} in column {key}, but a population target has been set"
                    )

    def validate_output(
        self, weights: np.ndarray, min_weight: float = 0.25, max_weight: int = 4
    ) -> None:
        # On average, weights should be very close to 1
        if not np.isclose(weights.mean(), 1, atol=1e-3):
            self.success = False
            raise WeightsConvergenceError("There was an issue with weight convergence")
        # Look for extreme weights
        if min_weight >= weights.min() or max_weight <= weights.max():
            if self.raise_on_extreme:
                self.success = False
                raise ExtremeWeightsError(
                    "Some observations have been ascribed extreme weight factors. "
                    f"Smallest weight: {weights.min()}, Largest weight: {weights.max()}"
                )
            else:
                extreme_weights(
                    "Some observations have been ascribed extreme weight factors. "
                    f"Smallest weight: {weights.min()}, Largest weight: {weights.max()}. "
                    f"Extreme weights will be clipped to range {self.cutoffs['lo']} thru {self.cutoffs['hi']}"
                )

    def clip_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Clips the weight factors to the specified lower and upper bounds.

        Parameters:
        - weights (numpy.ndarray): 1D array of weight factors.
        - cutoffs (dict): Dictionary with 'lo' as the lower bound and 'hi' as the upper bound.

        Returns:
        - numpy.ndarray: Clipped weight factors.
        """
        if "lo" not in self.cutoffs or "hi" not in self.cutoffs:
            raise ValueError("Cutoffs dictionary must contain 'lo' and 'hi' keys.")

        lower_bound = self.cutoffs["lo"]
        upper_bound = self.cutoffs["hi"]

        clipped_weights = np.clip(weights, lower_bound, upper_bound)
        return clipped_weights

    def rake(
        self,
        data: pd.DataFrame,
        max_steps: int = 2000,
        tolerance: float = 1e-6,
        early_stopping: int = 25,
    ) -> np.typing.NDArray[np.float64]:
        """Run the generalized raking algorithm

        Args:
            data: pd.DataFrame
                The survey dataset
            max_steps: int
                The maximum number of iterations to try and reach convergence
            tolerance: float
                Maximum tolerance for loss, convergence is reached if the loss is smaller than this value
            early_stopping: int
                Maximum number of iterations without improvement in loss

        Raises:
            WeightsConvergenceError if the algorithm did not converge before `max_steps`
                was reached or if `early_stopping` has been triggered

        Returns:
            Weight factors as a 1d NumPy array of floats
        """

        if self.exclusion_column:
            self.validate_exclusion_column(data)
            original_data = data
            data = original_data[original_data[self.exclusion_column] == 1]

        self.validate_input(data)

        design_matrix = self.create_design_matrix(data)
        target_vector = self.create_target_vector(data)
        num_respondents, num_targets = design_matrix.shape
        weights = np.ones(num_respondents)
        lagrange_multipliers = np.zeros(num_targets)
        identity_matrix = np.eye(num_respondents)
        early_stop_trigger = early_stopping

        for step in range(max_steps):
            # Update the Lagrange multipliers based on the current weights and constraints
            lagrange_multipliers += np.dot(
                np.linalg.pinv(
                    np.dot(np.dot(design_matrix.T, identity_matrix), design_matrix)
                ),
                (target_vector - np.dot(design_matrix.T, weights)),
            )
            # Update the weights vector by exponentiating the weighted sums of variables
            weights = np.exp(np.dot(design_matrix, lagrange_multipliers))
            # Update the diagonal scaling matrix based on the new weights
            identity_matrix = np.diag(weights)

            # Calculate loss
            old_loss = self.loss[-1]
            new_loss = np.max(
                np.abs(np.dot(design_matrix.T, weights) - target_vector) / target_vector
            )
            self.loss.append(round(new_loss, 4))

            # Check for convergence
            if new_loss < tolerance:
                self.success = True
                break
            else:
                if old_loss <= new_loss:
                    early_stop_trigger -= 1
                else:
                    early_stop_trigger = early_stopping
                if early_stop_trigger <= 0:
                    raise WeightsConvergenceError(
                        f"No reduction in loss for {early_stopping} iterations. Stopping. "
                        f"Total iterations: {step + 1}. Final loss: {self.loss[-1]:.4}"
                    )
        self.validate_output(weights)
        if not self.success:
            raise WeightsConvergenceError(
                f"Weights did not converge after {max_steps} iterations. "
                f"Final loss: {self.loss[-1]:.4}"
            )

        if not self.raise_on_extreme:
            weights = self.clip_weights(weights)

        if self.exclusion_column:
            # Create an array of 1.0 for rows with group_column equals 0
            non_weighted_indices = original_data.index.difference(data.index)

            # Combine the weights for selected rows and non-selected rows
            all_weights = np.ones(len(original_data))  # Initialize with ones
            all_weights[data.index] = weights  # Set weights for selected rows
            all_weights[non_weighted_indices] = 1.0  # Set weights for non-selected rows

            # Sort the weights array based on the original dataframe indices
            weights = all_weights[original_data.index.argsort()]

        self.weights = weights

        return weights
