from collections import defaultdict
from typing import DefaultDict, Dict, Callable

import torch


class MetricLogger:
    def __init__(
        self,
        window_size: int = 100,
        quiet: bool = False,
        log_fn: Callable[[str], None] = print,
        metric_name_prefix: str = "",
        aml_run=None,
    ):
        self._log_fn = log_fn
        self._metric_name_prefix = metric_name_prefix
        self._window_size = window_size
        self._quiet = quiet
        self._aml_run = aml_run

        self._step_counter = 0
        self._metrics: DefaultDict[str, float] = defaultdict(lambda: 0.0)
        self._windowed_metrics: DefaultDict[str, float] = defaultdict(lambda: 0.0)

    @staticmethod
    def __format_metric_dict(metric_dict: Dict[str, float], num_steps: int) -> str:
        return ", ".join(
            f"{metric_name}: {metric_val/num_steps:.5f}"
            for metric_name, metric_val in metric_dict.items()
        )

    def get_mean_metric_value(self, metric_name: str) -> float:
        return self._metrics[metric_name] / self._step_counter

    @property
    def metric_overview(self):
        return self.__format_metric_dict(self._metrics, self._step_counter)

    def log_metrics(self, **kwargs) -> None:
        for metric_name, metric_val in kwargs.items():
            if isinstance(metric_val, torch.Tensor):
                metric_val = metric_val.detach().cpu().item()
            self._metrics[metric_name] += metric_val
            self._windowed_metrics[metric_name] += metric_val

        self._step_counter += 1
        if self._step_counter % self._window_size == 0:
            if not self._quiet:
                mean_metric_str = self.__format_metric_dict(self._metrics, self._step_counter)
                windowed_metric_str = self.__format_metric_dict(
                    self._windowed_metrics, self._window_size
                )
                self._log_fn(
                    f" Step {self._step_counter:04d}"
                    f" || Mean metrics so far: {mean_metric_str}"
                    f" || This window: {windowed_metric_str}"
                )
            if self._aml_run is not None:
                for metric_name, metric_val in self._windowed_metrics.items():
                    self._aml_run.log(
                        f"{self._metric_name_prefix}{metric_name}", metric_val / self._window_size
                    )
            self._windowed_metrics.clear()
