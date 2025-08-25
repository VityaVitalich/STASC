"""Utilities for flatten dict."""

from dataclasses import asdict, is_dataclass
from typing import Any, List, Union

from omegaconf import ListConfig

from config import Config


def _process_values(value: Any) -> dict[str, str | int | float]:
    if value is None:
        return {"": "None"}
    if isinstance(value, bool):
        return {"": str(value)}
    if isinstance(value, (str, int, float)):
        return {"": value}
    if isinstance(value, (list, ListConfig)):
        return {
            f"{i}{k0}": v0 for i, v in enumerate(value) for k0, v0 in _process_values(v).items()
        }
    try:
        return {
            f"_{key}{k0}": v0 for key, v in value.items() for k0, v0 in _process_values(v).items()
        }
    except AttributeError as e:
        raise ValueError(f"Can not convert value {value} to scalar. Type is {type(value)}") from e


def flatten_dict(config: dict | Config | Any) -> dict[str, str | int | float]:
    """Convert a nested config to a flat dictionary suitable for TensorBoard logging.

    :param config: A dict or Config instance
    :return: A flat dictionary with keys of the form
        "outer_middle_inner".
    :raises ValueError: If values of config are not of type str, int,
        float or bool.
    """
    # Convert Config to a dictionary if it is a dataclass
    if is_dataclass(config):
        config = asdict(config)  # type: ignore
    elif not isinstance(config, dict):
        try:
            config = dict(config)
        except TypeError:
            raise ValueError("The provided config must be a dictionary or a dataclass.") from None

    # Flatten the dictionary
    return {
        f"{key}{k}": v for key, value in config.items() for k, v in _process_values(value).items()
    }


def flatten_predictions(predictions: Union[Any, List[Any]]) -> List[Any]:
    stack, result = [predictions], []
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(reversed(current))
        else:
            result.append(current)
    return result
