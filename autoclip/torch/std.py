from typing import Iterator, List, Dict, Union, Any
import torch

from autoclip.torch.clipper import Clipper, OptimizerWithClipping


class StandardClip(Clipper):
    def __init__(
        self,
        parameters: Iterator[torch.nn.parameter.Parameter],
        deviations: float = 2.0,
        history_length: int = 1000,
        global_threshold: bool = False,
    ) -> None:
        self.global_threshold = global_threshold
        self.global_deviations = None
        self.global_history_length = None
        if self.global_threshold:
            self.global_deviations = deviations
            self.global_history_length = history_length

        super().__init__(
            parameters,
            {"deviations": deviations, "history_length": history_length},
        )

    @classmethod
    def as_optimizer(
        cls: "StandardClip",
        optimizer: torch.optim.Optimizer,
        deviations: float = 2.0,
        history_length: int = 1000,
        global_threshold: bool = False,
    ) -> "OptimizerWithClipping":
        return super().as_optimizer(
            optimizer,
            deviations=deviations,
            history_length=history_length,
            global_threshold=global_threshold,
        )

    def verify_parameter_settings(self, settings: Dict[str, Any]) -> None:
        quantile = settings["deviations"]
        history_length = settings["history_length"]
        if not isinstance(quantile, (int, float, torch.Tensor)):
            raise TypeError(
                "StandardClip deviations value must be an int, float or a tensor."
            )
        if not isinstance(history_length, int):
            raise TypeError("StandardClip history_length must be an int.")
        if quantile < 0.0:
            raise ValueError(
                "StandardClip deviations value must be greater than or equal to 0."
            )
        if history_length <= 0:
            raise ValueError("StandardClip history length must be greater than zero.")

    def step(self) -> None:
        if self.global_threshold:
            self._clip_global()
        else:
            self._clip_local()

    def _clip_local(self):
        for parameter_group in self.parameter_groups:
            group_deviations = parameter_group["deviations"]
            group_history_length = parameter_group["history_length"]

            for parameter in parameter_group["params"]:
                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                if len(state) == 0:
                    state["history"] = torch.Tensor([]).to(parameter.device)
                    threshold = torch.inf
                else:
                    std = torch.std(state["history"])
                    threshold = std * group_deviations
                new_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameter, max_norm=threshold
                )
                state["history"] = torch.hstack((state["history"], new_grad_norm))[
                    -group_history_length:
                ]

    def _clip_global(self):
        parameters = []
        for parameter_group in self.parameter_groups:
            parameters = parameters + parameter_group["params"]

        if len(self.state["global_history"]) == 0:
            # Assumes all parameters are on the same device
            self.state["global_history"] = torch.Tensor([]).to(parameters[0].device)
            threshold = torch.inf
        else:
            std = torch.std(self.state["global_history"])
            threshold = std * self.global_deviations
        new_grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=threshold)
        self.state["global_history"] = torch.hstack(
            (self.state["global_history"], new_grad_norm)
        )[-self.global_history_length :]

    def add_param_group(
        self,
        parameter_group: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        deviations: float = None,
        history_length: int = None,
    ) -> None:
        parameter_group_args = {}
        if deviations is not None:
            parameter_group_args["deviations"] = deviations
        if history_length is not None:
            parameter_group_args["history_length"] = history_length
        return super().add_param_group(parameter_group, **parameter_group_args)
