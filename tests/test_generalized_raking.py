from weightfactors import GeneralizedRaker
from weightfactors.utils.exceptions import (
    WeightsConvergenceError,
    ExtremeWeightsError,
    ExtremeWeightsWarning,
)

import random

import numpy as np
import pandas as pd
import pytest


random.seed(42)


def test_invalid_input():
    with pytest.raises(
        KeyError, match="There is no column DoesNotExist in the provided dataset"
    ):
        data = pd.DataFrame({"Gender": ["Male", "Male", np.nan, "Female"]})
        raker = GeneralizedRaker({"DoesNotExist": {1: 0.5, 2: 0.5}})
        raker.rake(data)
    with pytest.raises(ValueError, match="Column Gender contains missing values"):
        data = pd.DataFrame({"Gender": ["Male", "Male", np.nan, "Female"]})
        raker = GeneralizedRaker({"Gender": {"Male": 0.5, "Female": 0.5}})
        raker.rake(data)
    with pytest.raises(
        KeyError,
        match="There are observations for a value in 'Gender' that has not been mapped to a population target",
    ):
        data = pd.DataFrame({"Gender": ["Male", "Male", "Female"]})
        raker = GeneralizedRaker({"Gender": {"DoesNotExist": 0.5, "Female": 0.5}})
        raker.rake(data)
    with pytest.raises(
        ValueError, match="The sum of the population distribution for 'Gender' is not 1"
    ):
        data = pd.DataFrame({"Gender": ["Male", "Male", "Female"]})
        raker = GeneralizedRaker({"Gender": {"Male": 0.51, "Female": 0.5}})
        raker.rake(data)
    with pytest.raises(
        KeyError,
        match="There are observations for a value in 'Gender' that has not been mapped to a population target",
    ):
        data = pd.DataFrame({"Gender": ["Male", "Male", "Female", "Other"]})
        raker = GeneralizedRaker({"Gender": {"Male": 0.51, "Female": 0.49}})
        raker.rake(data)


def test_generalized_raking_no_convergence():
    data = pd.DataFrame(
        {
            "Gender": ["Male"] + ["Female" for _ in range(99)],
            "Region": [0 for _ in range(97)] + [1] + [2] + [3],
            "Age": [0 for _ in range(98)] + [1] + [2],
            "Educ": [0] + [1 for _ in range(98)] + [2],
        },
    )
    population = {
        "Gender": {"Male": 0.5, "Female": 0.5},
        "Region": {0: 0.099, 1: 0.208, 2: 0.456, 3: 0.237},
        "Age": {0: 0.27, 1: 0.32, 2: 0.41},
        "Educ": {0: 0.33, 1: 0.33, 2: 0.34},
    }

    raker = GeneralizedRaker(population_targets=population)
    with pytest.raises(WeightsConvergenceError):
        raker.rake(data)


def test_generalized_raking_extreme_weights_raise():
    data = pd.DataFrame(
        {
            "Gender": ["Male" for _ in range(5)] + ["Female" for _ in range(95)],
        },
    )
    population = {
        "Gender": {"Male": 0.5, "Female": 0.5},
    }

    raker = GeneralizedRaker(population_targets=population, raise_on_extreme=True)
    with pytest.raises(ExtremeWeightsError):
        raker.rake(data)


def test_generalized_raking_extreme_weights_warn_and_clip():
    data = pd.DataFrame(
        {
            "Gender": ["Male" for _ in range(5)] + ["Female" for _ in range(95)],
        },
    )
    population = {
        "Gender": {"Male": 0.5, "Female": 0.5},
    }

    raker = GeneralizedRaker(population_targets=population)
    with pytest.warns(ExtremeWeightsWarning):
        weights = raker.rake(data)

    assert max(weights) == 4
    assert min(weights) == pytest.approx(0.5263157894736838)


def test_generalized_raking_one_column_simple():
    data = pd.DataFrame(
        {"Gender": ["Male" for _ in range(40)] + ["Female" for _ in range(60)]}
    )
    population = {"Gender": {"Male": 0.5, "Female": 0.5}}

    raker = GeneralizedRaker(population_targets=population)
    weights = raker.rake(data)
    assert weights.mean() == pytest.approx(1)
    assert weights[0] > weights[-1]

    X = raker.create_design_matrix(data)
    weighted_n = np.sum(X[:, 0] * weights) + np.sum(X[:, 1] * weights)
    new_male_proportion = np.sum(X[:, 0] * weights) / weighted_n
    new_female_proportion = np.sum(X[:, 1] * weights) / weighted_n
    assert new_male_proportion == pytest.approx(0.5, abs=1e-6)
    assert new_female_proportion == pytest.approx(0.5, abs=1e-6)


def test_generalized_raking_exclusion_column():
    data = pd.DataFrame(
        {
            "Gender": ["Male" for _ in range(40)] + ["Female" for _ in range(60)],
            "exclude": [1 for _ in range(75)] + [0 for _ in range(25)],
        }
    )
    population = {"Gender": {"Male": 0.5, "Female": 0.5}}

    raker = GeneralizedRaker(population_targets=population, exclusion_column="exclude")
    weights = raker.rake(data)
    assert sum(weights[-25:]) == 25
    weights = weights[:75]
    assert weights.mean() == pytest.approx(1)

    X = raker.create_design_matrix(data[:75])
    weighted_n = np.sum(X[:, 0] * weights) + np.sum(X[:, 1] * weights)
    new_male_proportion = np.sum(X[:, 0] * weights) / weighted_n
    new_female_proportion = np.sum(X[:, 1] * weights) / weighted_n
    assert new_male_proportion == pytest.approx(0.5, abs=1e-6)
    assert new_female_proportion == pytest.approx(0.5, abs=1e-6)


def test_generalized_raking_one_column_harder():
    data = pd.DataFrame(
        {
            "Region": [0 for _ in range(47)]
            + [1 for _ in range(201)]
            + [2 for _ in range(460)]
            + [3 for _ in range(292)]
        }
    )
    population = {"Region": {0: 0.099, 1: 0.208, 2: 0.456, 3: 0.237}}

    raker = GeneralizedRaker(population_targets=population)
    weights = raker.rake(data)
    assert weights.mean() == pytest.approx(1)
    assert weights[0] > weights[-1]

    X = raker.create_design_matrix(data)
    weighted_n = (
        np.sum(X[:, 0] * weights)
        + np.sum(X[:, 1] * weights)
        + np.sum(X[:, 2] * weights)
        + np.sum(X[:, 3] * weights)
    )
    weighted_north_proportion = np.sum(X[:, 0] * weights) / weighted_n
    new_east_proportion = np.sum(X[:, 1] * weights) / weighted_n
    new_west_proportion = np.sum(X[:, 2] * weights) / weighted_n
    new_south_proportion = np.sum(X[:, 3] * weights) / weighted_n
    assert weighted_north_proportion == pytest.approx(0.099, abs=1e-6)
    assert new_east_proportion == pytest.approx(0.208, abs=1e-6)
    assert new_west_proportion == pytest.approx(0.456, abs=1e-6)
    assert new_south_proportion == pytest.approx(0.237, abs=1e-6)


def test_generalized_raking_two_columns():
    gender = [0 for _ in range(400)] + [1 for _ in range(600)]
    random.shuffle(gender)
    region = (
        [0 for _ in range(67)]
        + [1 for _ in range(201)]
        + [2 for _ in range(490)]
        + [3 for _ in range(242)]
    )
    random.shuffle(region)
    data = pd.DataFrame({"Gender": gender, "Region": region})
    population = {
        "Gender": {0: 0.49, 1: 0.51},
        "Region": {0: 0.099, 1: 0.208, 2: 0.456, 3: 0.237},
    }

    raker = GeneralizedRaker(population_targets=population)
    weights = raker.rake(data)
    assert weights.mean() == pytest.approx(1)

    X = raker.create_design_matrix(data)
    gender_n = np.sum(X[:, 0] * weights) + np.sum(X[:, 1] * weights)

    region_n = (
        np.sum(X[:, 2] * weights)
        + np.sum(X[:, 3] * weights)
        + np.sum(X[:, 4] * weights)
        + np.sum(X[:, 5] * weights)
    )
    new_male_proportion = np.sum(X[:, 0] * weights) / gender_n
    new_female_proportion = np.sum(X[:, 1] * weights) / gender_n
    weighted_north_proportion = np.sum(X[:, 2] * weights) / region_n
    new_east_proportion = np.sum(X[:, 3] * weights) / region_n
    new_west_proportion = np.sum(X[:, 4] * weights) / region_n
    new_south_proportion = np.sum(X[:, 5] * weights) / region_n
    assert new_male_proportion == pytest.approx(0.49, abs=1e-6)
    assert new_female_proportion == pytest.approx(0.51, abs=1e-6)
    assert weighted_north_proportion == pytest.approx(0.099, abs=1e-6)
    assert new_east_proportion == pytest.approx(0.208, abs=1e-6)
    assert new_west_proportion == pytest.approx(0.456, abs=1e-6)
    assert new_south_proportion == pytest.approx(0.237, abs=1e-6)


def test_generalized_raking_three_columns():
    gender = [0 for _ in range(400)] + [1 for _ in range(600)]
    random.shuffle(gender)
    region = (
        [0 for _ in range(90)]
        + [1 for _ in range(190)]
        + [2 for _ in range(470)]
        + [3 for _ in range(250)]
    )
    random.shuffle(region)
    age = [0 for _ in range(280)] + [1 for _ in range(330)] + [2 for _ in range(390)]
    random.shuffle(age)

    data = pd.DataFrame(
        {
            "Age": age,
            "Gender": gender,
            "Region": region,
        }
    )
    population = {
        "Age": {0: 0.27, 1: 0.32, 2: 0.41},
        "Gender": {0: 0.49, 1: 0.51},
        "Region": {0: 0.099, 1: 0.208, 2: 0.456, 3: 0.237},
    }

    raker = GeneralizedRaker(population_targets=population)
    weights = raker.rake(data)
    assert weights.mean() == pytest.approx(1)

    X = raker.create_design_matrix(data)
    age_n = (
        np.sum(X[:, 0] * weights)
        + np.sum(X[:, 1] * weights)
        + np.sum(X[:, 2] * weights)
    )
    gender_n = np.sum(X[:, 3] * weights) + np.sum(X[:, 4] * weights)

    region_n = (
        np.sum(X[:, 5] * weights)
        + np.sum(X[:, 6] * weights)
        + np.sum(X[:, 7] * weights)
        + np.sum(X[:, 8] * weights)
    )

    new_young_proportion = np.sum(X[:, 0] * weights) / age_n
    new_middle_proportion = np.sum(X[:, 1] * weights) / age_n
    new_old_proportion = np.sum(X[:, 2] * weights) / age_n
    new_male_proportion = np.sum(X[:, 3] * weights) / gender_n
    new_female_proportion = np.sum(X[:, 4] * weights) / gender_n
    weighted_north_proportion = np.sum(X[:, 5] * weights) / region_n
    new_east_proportion = np.sum(X[:, 6] * weights) / region_n
    new_west_proportion = np.sum(X[:, 7] * weights) / region_n
    new_south_proportion = np.sum(X[:, 8] * weights) / region_n
    assert new_young_proportion == pytest.approx(0.27, abs=1e-6)
    assert new_middle_proportion == pytest.approx(0.32, abs=1e-6)
    assert new_old_proportion == pytest.approx(0.41, abs=1e-6)
    assert new_male_proportion == pytest.approx(0.49, abs=1e-6)
    assert new_female_proportion == pytest.approx(0.51, abs=1e-6)
    assert weighted_north_proportion == pytest.approx(0.099, abs=1e-6)
    assert new_east_proportion == pytest.approx(0.208, abs=1e-6)
    assert new_west_proportion == pytest.approx(0.456, abs=1e-6)
    assert new_south_proportion == pytest.approx(0.237, abs=1e-6)
