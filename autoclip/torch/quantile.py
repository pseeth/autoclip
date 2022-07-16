from typing import Iterator, List, Dict, Union
import torch

from autoclip.torch.clipper import Clipper, OptimizerWithClipping


class QuantileClip(Clipper):
    def __init__(
        self,
        parameters: Iterator[torch.nn.parameter.Parameter],
        quantile: float = 0.9,
        history_length: int = 1000,
        global_threshold: bool = False,
    ) -> None:
        self.global_threshold = global_threshold
        self.global_quantile = None
        self.global_history_length = None
        if self.global_threshold:
            self.global_quantile = quantile
            self.global_history_length = history_length

        super().__init__(
            parameters,
            {"quantile": quantile, "history_length": history_length},
        )

    @classmethod
    def as_optimizer(
        cls: "QuantileClip",
        optimizer: torch.optim.Optimizer,
        quantile: float = 0.9,
        history_length: int = 1000,
        global_threshold: bool = False,
    ) -> "OptimizerWithClipping":
        return super().as_optimizer(
            optimizer,
            quantile=quantile,
            history_length=history_length,
            global_threshold=global_threshold,
        )

    def step(self) -> None:
        if self.global_threshold:
            self._clip_global()
        else:
            self._clip_local()

    def _clip_local(self):
        for parameter_group in self.parameter_groups:
            group_quantile = parameter_group["quantile"]
            group_history_length = parameter_group["history_length"]

            for parameter in parameter_group["params"]:
                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                if len(state) == 0:
                    state["history"] = torch.Tensor([]).to(parameter.device)
                    threshold = torch.inf
                else:
                    threshold = torch.quantile(state["history"], group_quantile)
                new_grad_norm = torch.nn.utils.clip_grad_norm(
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
            threshold = torch.quantile(
                self.state["global_history"], self.global_quantile
            )
        new_grad_norm = torch.nn.utils.clip_grad_norm(parameters, max_norm=threshold)
        self.state["global_history"] = torch.hstack(
            (self.state["global_history"], new_grad_norm)
        )[-self.global_history_length :]

    def add_param_group(
        self,
        parameter_group: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        quantile: float = None,
        history_length: int = None,
    ) -> None:
        parameter_group_args = {}
        if quantile is not None:
            parameter_group_args["quantile"] = quantile
        if history_length is not None:
            parameter_group_args["history_length"] = history_length
        return super().add_param_group(parameter_group, **parameter_group_args)
