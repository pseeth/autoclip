from typing import Iterator, List, Dict, Union
import torch

from autoclip.torch.clipper import Clipper


class StandardClip(Clipper):
    def __init__(
        self,
        parameters: Iterator[torch.nn.parameter.Parameter],
        clip_deviations: float = 2.0,
        history_length: int = 1000,
        global_threshold: bool = False,
    ) -> None:
        self.global_threshold = global_threshold
        self.global_clip_deviations = None
        self.global_history_length = None
        if self.global_threshold:
            self.global_clip_deviations = clip_deviations
            self.global_history_length = history_length

        super().__init__(
            parameters,
            {"clip_deviations": clip_deviations, "history_length": history_length},
        )

    def step(self) -> None:
        if self.global_threshold:
            self._clip_global()
        else:
            self._clip_local()

    def _clip_local(self):
        for parameter_group in self.parameter_groups:
            group_clip_deviations = parameter_group["clip_deviations"]
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
                    threshold = std * group_clip_deviations
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
            std = torch.std(self.state["global_history"])
            threshold = std * self.global_clip_deviations
        new_grad_norm = torch.nn.utils.clip_grad_norm(parameters, max_norm=threshold)
        self.state["global_history"] = torch.hstack(
            (self.state["global_history"], new_grad_norm)
        )[-self.global_history_length :]

    def add_param_group(
        self,
        parameter_group: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        clip_deviations: float = None,
        history_length: int = None,
    ) -> None:
        parameter_group_args = {}
        if clip_deviations is not None:
            parameter_group_args["clip_deviations"] = clip_deviations
        if history_length is not None:
            parameter_group_args["history_length"] = history_length
        return super().add_param_group(parameter_group, **parameter_group_args)
