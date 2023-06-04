# stdlib
from typing import Any, Dict, List

# third party
import optuna

# synthcity absolute
import synthcity.plugins.core.distribution as D
from synthcity.utils.callbacks import Callback, ValidationMixin


def suggest(trial: optuna.Trial, dist: D.Distribution) -> Any:
    if isinstance(dist, D.FloatDistribution):
        return trial.suggest_float(dist.name, dist.low, dist.high)
    elif isinstance(dist, D.LogDistribution):
        return trial.suggest_float(dist.name, dist.low, dist.high, log=True)
    elif isinstance(dist, D.IntegerDistribution):
        return trial.suggest_int(dist.name, dist.low, dist.high, dist.step)
    elif isinstance(dist, D.IntLogDistribution):
        return trial.suggest_int(dist.name, dist.low, dist.high, log=True)
    elif isinstance(dist, D.CategoricalDistribution):
        return trial.suggest_categorical(dist.name, dist.choices)
    else:
        raise ValueError(f"Unknown distribution: {dist}")


def suggest_all(trial: optuna.Trial, distributions: List[D.Distribution]) -> Dict:
    return {dist.name: suggest(trial, dist) for dist in distributions}


class OptunaPruning(Callback):
    def __init__(self, trial: optuna.Trial) -> None:
        self.trial = trial
        self._steps = 0

    def on_fit_begin(self, model: ValidationMixin) -> None:
        pass

    def on_fit_end(self, model: ValidationMixin) -> None:
        pass

    def on_epoch_begin(self, model: ValidationMixin) -> None:
        pass

    def on_epoch_end(self, model: ValidationMixin) -> None:
        self.trial.report(model.valid_score, self._steps)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        self._steps += 1
