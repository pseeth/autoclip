from typing import Union, Mapping, Collection
import torch


def deep_tensor_move(
    tensors: Union[Mapping, Collection, torch.Tensor], device: Union[torch.device, str]
) -> Union[Mapping, Collection, torch.Tensor]:
    """
    Extracted from torch.optim.Optimizer's load_state_dict method.
    """
    if isinstance(tensors, torch.Tensor):
        tensors = tensors.to(device=device)
        return tensors
    elif isinstance(tensors, Mapping):
        return type(tensors)(
            {
                key: deep_tensor_move(tensors=value, device=device)
                for key, value in tensors.items()
            }
        )
    elif isinstance(tensors, Collection) and not isinstance(tensors, str):
        return type(tensors)(
            [deep_tensor_move(tensors=value, device=device) for value in tensors]
        )
    else:
        return tensors
