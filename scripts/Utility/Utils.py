import numpy as np
import torch
from typing import TypeVar, Sequence, Callable, Optional, Literal
from typing_extensions import ParamSpec
from functools import wraps
from dataclasses import is_dataclass, fields

I = TypeVar("I")
T_In = ParamSpec('T_In')
T_Out = TypeVar('T_Out')


def reflect_torch_dtype(type_string: Literal["fp32", "bf16", "fp16"]) -> torch.dtype:
    if type_string == "fp32":
        return torch.float32
    elif type_string == "bf16":
        return torch.bfloat16
    elif type_string == "fp16":
        return torch.float16
    else:
        raise ValueError(f"Expect to be one of fp32/bf16/fp16, but got `{type_string}`")


def StructuralMove(obj, device) -> Optional[torch.Tensor] or list or dict:
    if torch.is_tensor(obj):
        return obj.to(device)
    elif obj is None:
        return None
    elif isinstance(obj, dict):
        return {k: StructuralMove(obj[k], device) for k in obj}
    elif isinstance(obj, list):
        return [StructuralMove(v, device) for v in obj]
    elif isinstance(obj, np.ndarray):
        return torch.tensor(obj).to(device)
    else:
        raise ValueError(f"Unable to move type {type(obj)}")


def cropToMultiple(x: torch.Tensor, factor: int or list, dim: int or list):
    def _cropToMultiple(x: torch.Tensor, factor: int, dim: int):
        size = x.size(dim)
        to_crop = (size % factor) // 2
        if to_crop == 0:
            return x
        return x.narrow(dim, to_crop, size - (2 * to_crop))

    if isinstance(factor, int) and isinstance(dim, int):
        return _cropToMultiple(x, factor, dim)
    elif isinstance(factor, int) and isinstance(dim, list):
        result = x
        for d in dim:
            result = _cropToMultiple(result, factor, d)
        return result
    elif isinstance(factor, list) and isinstance(dim, list):
        result = x
        for f, d in zip(factor, dim):
            result = _cropToMultiple(result, f, d)
        return result
    else:
        raise ValueError("Unexpected combination for cropToMultiple - only accept [list, list], [int, list] and [int, int]")


def centerCropTo(x: torch.Tensor, shape: int or list, dim: int or list):
    def _centerCropTo(x: torch.Tensor, shape: int, dim: int):
        size = x.size(dim)
        to_crop = (size - shape) // 2
        if to_crop == 0:
            return x
        return x.narrow(dim, to_crop, shape)

    if isinstance(shape, int) and isinstance(dim, int):
        return _centerCropTo(x, shape, dim)
    elif isinstance(shape, int) and isinstance(dim, list):
        result = x
        for d in dim:
            result = _centerCropTo(result, shape, d)
        return result
    elif isinstance(shape, list) and isinstance(dim, list):
        result = x
        for f, d in zip(shape, dim):
            result = _centerCropTo(result, f, d)
        return result
    else:
        raise ValueError(
            "Unexpected combination for centerCropTo - only accept [list, list], [int, list] and [int, int]"
        )


def padTo(x: torch.Tensor, sizes: int or Sequence[int], dim: int or Sequence[int], value: float):
    def _padTo(x: torch.Tensor, factor: int, dim: int, pad_value: float):
        size = x.size(dim)
        assert (factor - size) % 2 == 0, f"Can only handle even padding. Target_size={factor}, Actual_size={size} on dim {dim}."
        to_pad = (factor - size) // 2
        if to_pad == 0:
            return x

        pad_config = [0] * (x.dim() * 2)
        pad_config[-2 * (dim + 1)] = to_pad
        pad_config[-2 * (dim + 1) + 1] = to_pad
        pad_config = tuple(pad_config)
        return torch.nn.functional.pad(x, pad_config, mode="constant", value=pad_value)

    if isinstance(sizes, int) and isinstance(dim, int):
        return _padTo(x, sizes, dim, value)
    elif isinstance(sizes, int) and isinstance(dim, (list, tuple)):
        result = x
        for d in dim:
            result = _padTo(result, sizes, d, value)
        return result
    elif isinstance(sizes, (list, tuple)) and isinstance(dim, (list, tuple)):
        result = x
        for f, d in zip(sizes, dim):
            result = _padTo(result, f, d, value)
        return result
    else:
        raise ValueError(
            "Unexpected combination for padTo - only accept [list, list], [int, list] and [int, int]"
        )


def getConsecutiveRange(values: Sequence[I], pred: Callable[[I], bool]) -> list[tuple[int, int]]:
    ranges, start = [], -1
    for idx, item in enumerate(values):
        if pred(item) and start == -1:
            start = idx
        elif not pred(item) and start != -1:
            ranges.append((start, idx))
            start = -1
    return ranges


def IgnoreException(func: Callable[T_In, T_Out]) -> Callable[T_In, Optional[T_Out]]:
    @wraps(func)
    def wrapped(*args, **kwargs) -> Optional[T_Out]:
        try:
            result = func(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # Logger.show_exception()
            # Logger.write("warn", f"Failed to execute function {func.__name__} due to exception:\n{e}")
            result = None
        return result
    return wrapped


def tensor_safe_asdict(obj):
    if is_dataclass(obj):
        return {f.name: tensor_safe_asdict(getattr(obj, f.name)) for f in fields(obj)}
    elif isinstance(obj, dict):
        return {k: tensor_safe_asdict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(tensor_safe_asdict(v) for v in obj)
    elif isinstance(obj, torch.Tensor):
        return obj
    else:
        return obj
