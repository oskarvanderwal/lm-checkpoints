"""Utility functions for lm-checkpoints."""

from typing import List, Dict, Union


def records_to_list(list_of_dicts: Union[List[Dict[str, int]], Dict[str, int]]):
    """Transform a list of dictionaries to a dictionary of lists.

    Args:
        list_of_dicts: List of dictionaries, assuming each dictionary has the same keys.

    Returns:
        Dictionary of lists.
    """
    if not isinstance(list_of_dicts, list):
        list_of_dicts = [list_of_dicts]
    return {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}


def chunk(items: list, n: int) -> list:
    """Partition items into n chunks.

    Chunks differ in size by at most one element.

    >>> chunk(['a', 'b', 'c', 'd'], 2)
    [['a', 'b'], ['c', 'd']]
    >>> chunk(['a', 'b', 'c', 'd'], 3)
    [['a'], ['b', 'c'], ['d']]
    """
    size = len(items) / float(n)

    def idx(i):
        return int(round(i))

    return [items[idx(size * i) : idx(size * (i + 1))] for i in range(n)]


def nearest_available_step(
    tokens: int,
    tokens_per_step: int,
    available_steps: List[int],
) -> int:
    """Convert tokens to the nearest available training step.

    Args:
        tokens: Number of tokens seen.
        tokens_per_step: Tokens processed per training step.
        available_steps: List of available checkpoint steps.

    Returns:
        Nearest available step, clamped to valid range.
    """
    if not available_steps:
        raise ValueError("available_steps cannot be empty")

    target_step = tokens // tokens_per_step
    return min(available_steps, key=lambda s: abs(s - target_step))


def nearest_step_with_interval(
    tokens: int,
    tokens_per_step: int,
    min_step: int,
    max_step: int,
    interval: int = 1000,
) -> int:
    """Convert tokens to nearest step, rounding to interval and clamping to range.

    Args:
        tokens: Number of tokens seen.
        tokens_per_step: Tokens processed per training step.
        min_step: Minimum available step.
        max_step: Maximum available step.
        interval: Step interval (e.g., 1000 for checkpoints every 1000 steps).

    Returns:
        Nearest step rounded to interval, clamped to [min_step, max_step].
    """
    target_step = tokens // tokens_per_step
    # Round to nearest interval
    target_step = round(target_step / interval) * interval
    # Clamp to valid range
    return max(min_step, min(target_step, max_step))
