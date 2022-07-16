import torch

from autoclip.torch.utils import deep_tensor_move


def test_deep_tensor_move_dicts():
    structure = {
        "some_value": torch.rand((10, 10)),
        "another_value": {"some_nested_thing": torch.rand(5)},
    }
    deep_tensor_move(structure, "cpu")
    deep_tensor_move(structure, torch.device("cpu"))


def test_deep_tensor_move_lists():
    structure = [torch.rand((6, 12, 2)), [torch.rand(5), torch.rand(10, 4)]]
    deep_tensor_move(structure, "cpu")
    deep_tensor_move(structure, torch.device("cpu"))


def test_deep_tensor_move_tuples():
    structure = (torch.rand((6, 12, 2)), (torch.rand(5), torch.rand(10, 4)))
    deep_tensor_move(structure, "cpu")
    deep_tensor_move(structure, torch.device("cpu"))


def test_deep_tensor_move_tensors():
    structure = torch.rand((1, 2, 1, 6, 3))
    deep_tensor_move(structure, "cpu")
    deep_tensor_move(structure, torch.device("cpu"))


def test_deep_tensor_move_non_tensors():
    structure = {
        "value": 1.0,
        "list": [
            "string",
            {
                "some-value": torch.rand((6, 7)),
            },
        ],
    }
    deep_tensor_move(structure, "cpu")
    deep_tensor_move(structure, torch.device("cpu"))
