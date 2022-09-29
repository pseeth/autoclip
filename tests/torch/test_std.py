import torch
import pytest
from autoclip.torch import StandardClip
from utilities.torch import (
    run_clipper_initialization_error_tests,
    run_clipping_test,
    run_clipping_test_wrapper,
    run_add_param_group,
)


def test_create_clipper():
    example_model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    clipper = StandardClip(example_model.parameters())
    clipper = StandardClip(
        example_model.parameters(), deviations=0.5, history_length=500
    )
    clipper = StandardClip(example_model.parameters(), deviations=0.0)
    clipper = StandardClip(example_model.parameters(), deviations=1.0)


def test_clipper_parameters_not_parameters():
    with pytest.raises(TypeError):
        clipper = StandardClip(1.0)

    with pytest.raises(TypeError):
        clipper = StandardClip(torch.nn.Linear(10, 10))


def test_bad_history_values():
    run_clipper_initialization_error_tests(
        StandardClip,
        "history_length",
        [-1.0, 5.0, 0, -100],
        [TypeError, TypeError, ValueError, ValueError],
    )


def test_bad_deviations_values():
    run_clipper_initialization_error_tests(
        StandardClip, "deviations", [-1.0, -0.5], [ValueError, ValueError]
    )


def test_create_optimizer_wrapper():
    example_model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    optimizer = torch.optim.AdamW(example_model.parameters())
    clipper = StandardClip.as_optimizer(optimizer=optimizer)
    clipper = StandardClip.as_optimizer(
        optimizer=optimizer, deviations=0.5, history_length=500
    )
    clipper = StandardClip.as_optimizer(optimizer=optimizer, deviations=0.0)
    clipper = StandardClip.as_optimizer(optimizer=optimizer, deviations=1.0)

    optimizer = torch.optim.LBFGS(example_model.parameters())
    clipper = StandardClip.as_optimizer(optimizer=optimizer)
    clipper = StandardClip.as_optimizer(
        optimizer=optimizer, deviations=0.5, history_length=500
    )
    clipper = StandardClip.as_optimizer(optimizer=optimizer, deviations=0.0)
    clipper = StandardClip.as_optimizer(optimizer=optimizer, deviations=1.0)


def test_clip_local():
    run_clipping_test(StandardClip, {})
    run_clipping_test(StandardClip, {"deviations": 0.5})
    run_clipping_test(StandardClip, {"history_length": 500})
    run_clipping_test(StandardClip, {"deviations": 0.5, "history_length": 500})


def test_clip_global():
    run_clipping_test(StandardClip, {"global_threshold": True})
    run_clipping_test(StandardClip, {"deviations": 0.5, "global_threshold": True})
    run_clipping_test(StandardClip, {"history_length": 500, "global_threshold": True})
    run_clipping_test(
        StandardClip,
        {"deviations": 0.5, "history_length": 500, "global_threshold": True},
    )


def test_clip_wrapper():
    run_clipping_test_wrapper(StandardClip, {})
    run_clipping_test_wrapper(StandardClip, {"deviations": 0.5})
    run_clipping_test_wrapper(StandardClip, {"history_length": 500})
    run_clipping_test_wrapper(StandardClip, {"deviations": 0.5, "history_length": 500})


def test_add_param_group():
    run_add_param_group(StandardClip, {})
    run_add_param_group(StandardClip, {"deviations": 0.5})
    run_add_param_group(StandardClip, {"history_length": 500})
    run_add_param_group(StandardClip, {"deviations": 0.5, "history_length": 500})


def test_save_state_dict():
    example_model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    clipper = StandardClip(example_model.parameters())
    state_dict = clipper.state_dict()
    clipper = StandardClip(example_model.parameters(), global_threshold=True)
    state_dict = clipper.state_dict()


def test_save_state_dict_wrapper():
    example_model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    optimizer = torch.optim.AdamW(example_model.parameters())
    clipper = StandardClip.as_optimizer(optimizer=optimizer)
    state_dict = clipper.state_dict()


def test_load_state_dict():
    example_model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    clipper = StandardClip(example_model.parameters())
    state_dict = clipper.state_dict()
    example_model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    clipper = StandardClip(example_model.parameters())
    clipper.load_state_dict(state_dict=state_dict)


def test_load_state_dict_wrapper():
    example_model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    optimizer = torch.optim.AdamW(example_model.parameters())
    clipper = StandardClip.as_optimizer(optimizer=optimizer)
    state_dict = clipper.state_dict()
    example_model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    optimizer = torch.optim.AdamW(example_model.parameters())
    clipper = StandardClip.as_optimizer(optimizer=optimizer)
    clipper.load_state_dict(state_dict=state_dict)


def test_clip_after_state_dict_load():
    example_model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    clipper = StandardClip(example_model.parameters())
    state_dict = clipper.state_dict()
    example_model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    clipper = StandardClip(example_model.parameters())
    clipper.load_state_dict(state_dict=state_dict)
    loss_fn = torch.nn.MSELoss()
    prediction = example_model(torch.rand((10)))
    target = torch.Tensor([10.0])
    loss = loss_fn(prediction, target)
    loss.backward()
    clipper.step()


def test_pickle_optimizer_wrapper():
    import io

    example_model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(example_model.parameters())
    clipper = StandardClip.as_optimizer(optimizer=optimizer)
    buffer = io.BytesIO()
    torch.save(clipper, buffer)
    buffer.seek(0)
    clipper = torch.load(buffer)
    clipper.optimizer


def test_pickle_clipper():
    import io

    example_model = torch.nn.Linear(10, 1)
    clipper = StandardClip(example_model.parameters())
    buffer = io.BytesIO()
    torch.save(clipper, buffer)
    buffer.seek(0)
    clipper = torch.load(buffer)
