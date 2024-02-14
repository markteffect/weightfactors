import warnings


class WeightsConvergenceError(Exception): ...


class ExtremeWeightsError(Exception): ...


class ExtremeWeightsWarning(Warning): ...


def extreme_weights(message: str) -> None:
    warnings.warn(ExtremeWeightsWarning(message), stacklevel=2)
