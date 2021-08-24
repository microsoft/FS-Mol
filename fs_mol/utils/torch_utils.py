import dataclasses

import numpy as np
import torch


def torchify(data, device: torch.device):
    if isinstance(data, (int, float, str, torch.Tensor)):
        return data
    elif isinstance(data, tuple):
        return tuple(torchify(e, device) for e in data)
    elif isinstance(data, list):
        return list(torchify(e, device) for e in data)
    elif isinstance(data, dict):
        return {k: torchify(v, device) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    elif dataclasses.is_dataclass(data):
        # Note that we can't use dataclasses.asdict, as this recursively turns
        # all values into dicts as well, so that we lose the inner structure...
        torch_data = {
            f.name: torchify(getattr(data, f.name), device) for f in dataclasses.fields(data)
        }
        return dataclasses.replace(data, **torch_data)
    else:
        raise ValueError(f"Trying to torchify unknown value type {type(data)}!")
