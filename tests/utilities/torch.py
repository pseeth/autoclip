import torch
import autoclip
import pytest
from autoclip.torch.clipper import Clipper
from autoclip.torch import QuantileClip, StandardClip


def run_clipper_initialization_error_tests(
    clipper_type: Clipper, value_name: str, bad_values: list, error_types: list
):
    example_model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    for bad_value, error_type in zip(bad_values, error_types):
        with pytest.raises(error_type) as _:
            clipper = clipper_type(
                example_model.parameters(), **{value_name: bad_value}
            )

        with pytest.raises(error_type):
            additional_parameters = torch.nn.Sequential(torch.nn.Linear(10, 10))
            clipper = clipper_type(example_model.parameters())
            clipper.add_param_group(
                additional_parameters.parameters(), **{value_name: bad_value}
            )


def run_clipping_test(clipper: Clipper, clipper_args: dict):
    example_model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    clipper = clipper(example_model.parameters(), **clipper_args)
    loss_fn = torch.nn.MSELoss()
    prediction = example_model(torch.rand((10)))
    target = torch.Tensor([10.0])
    loss = loss_fn(prediction, target)
    loss.backward()
    clipper.step()


def run_clipping_test_wrapper(clipper: Clipper, clipper_args: dict):
    example_model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(example_model.parameters())
    optimizer = clipper.as_optimizer(optimizer, **clipper_args)
    prediction = example_model(torch.rand((10)))
    target = torch.Tensor([10.0])
    loss = loss_fn(prediction, target)
    loss.backward()
    optimizer.step()


def run_add_param_group(clipper: Clipper, clipper_args: dict):
    example_model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    additional_model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    clipper = clipper(example_model.parameters())
    clipper.add_param_group(additional_model.parameters(), **clipper_args)
