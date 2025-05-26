import json
import re
import numpy as np


from hydra.utils import get_class
import inspect

import aiofiles
import torch
import torch.nn.functional as F


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    np.random.seed(seed)  # Also set numpy seed since we use numpy
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Ensure deterministic behavior


async def write_json(file_path, data):
    """
    Asynchronously writes data to a JSON file after filtering out non-serializable entries.

    Parameters:
    file_path (str): The path to the file where the JSON data will be stored.
    data (dict or list): The data to be serialized and written to file.

    Returns:
    None: This function does not return anything but writes to a file.
    """

    def is_json_serializable(value):
        try:
            json.dumps(value)
            return True
        except (TypeError, OverflowError):
            return False

    def filter_serializable(data):
        if isinstance(data, dict):
            return {k: filter_serializable(v) for k, v in data.items() if is_json_serializable(v)}
        elif isinstance(data, list):
            return [filter_serializable(item) for item in data if is_json_serializable(item)]
        else:
            return data if is_json_serializable(data) else None

    serializable_data = filter_serializable(data)

    async with aiofiles.open(file_path, "w") as file:
        await file.write(json.dumps(serializable_data, indent=4))


def sort_layer_ids(layer_ids):
    """Sort layer IDs by layer number, handling multiple naming formats

    Supports formats:
    - layer_X
    - layer_X.mlp
    - layers.X
    - layers.X.mlp
    - blocks.X.hook_resid_pre
    - blocks.X.hook_resid_post
    """

    def get_layer_num(layer_id):
        # Try all possible patterns
        patterns = [
            (r"layer_(\d+)", False),  # layer_X format
            (r"layers\.(\d+)", False),  # layers.X format
            (r"layer_(\d+)\.mlp", True),  # layer_X.mlp format
            (r"layers\.(\d+)\.mlp", True),  # layers.X.mlp format
            (r"blocks\.(\d+)\.hook_resid_pre", False),  # blocks.X.hook_resid_pre format
            (r"blocks\.(\d+)\.hook_resid_post", False),  # blocks.X.hook_resid_post format
        ]

        for pattern, is_mlp in patterns:
            match = re.search(pattern, layer_id)
            if match:
                layer_num = int(match.group(1))
                # Add .5 to MLP layers and post-residual hooks to maintain order within same layer
                return layer_num + (0.5 if is_mlp else 0)

        return float('inf')

    layer_ids = list(layer_ids)

    sorted_ids = sorted(layer_ids, key=get_layer_num)

    return sorted_ids


def get_required_init_args(cls):
    sig = inspect.signature(cls.__init__)
    return [
        name
        for name, param in sig.parameters.items()
        if name != 'self'
        and param.default == inspect.Parameter.empty
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]


def pearson_corrcoef(v1, v2):
    """
    Calculates the Pearson correlation coefficient between two vectors.

    Parameters:
    v1 (torch.Tensor): First input vector.
    v2 (torch.Tensor): Second input vector.

    Returns:
    float: Pearson correlation coefficient (-1 to 1 range), where 1 means total positive linear correlation,
           0 means no linear correlation, and -1 means total negative linear correlation.
    """
    v1_mean = v1 - v1.mean()
    v2_mean = v2 - v2.mean()
    norm_product = torch.norm(v1_mean, 2) * torch.norm(v2_mean, 2)
    return torch.dot(v1_mean, v2_mean) / norm_product if norm_product != 0 else 0


def angular_distance(v1, v2):
    """
    Computes the angular distance between two vectors, normalized to the range [0, 1].

    Parameters:
    v1 (torch.Tensor): First vector.
    v2 (torch.Tensor): Second vector.

    Returns:
    float: Angular distance between the vectors, scaled between 0 (no angular distance) and 1 (opposite directions).
    """
    cos_sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
    angle_rad = torch.acos(cos_sim.clamp(-1, 1))  # Clamp for numerical stability
    return angle_rad / torch.pi  # Normalize by pi to scale between 0 and 1


def basic_pattern_matching(v1, v2, threshold=0.1):
    """
    Calculates the pattern matching score between two vectors by counting
    the proportion of indices where non-zero elements are within a specified threshold.
    This version ignores zero-to-zero matches to focus on informative parts of the vectors.

    Parameters:
    v1 (torch.Tensor): First vector.
    v2 (torch.Tensor): Second vector.
    threshold (float): Maximum difference between non-zero elements to be considered close.

    Returns:
    float: Proportion of close indices among non-zero elements (0 to 1 range), where 1 indicates all are close.
    """
    # Mask to identify non-zero elements in both vectors
    non_zero_mask = (v1 != 0) & (v2 != 0)

    # Apply mask to both vectors
    v1_non_zero = v1[non_zero_mask]
    v2_non_zero = v2[non_zero_mask]

    if len(v1_non_zero) == 0:
        return 0.0  # Handle case with no non-zero comparisons to avoid division by zero

    # Calculate differences for non-zero elements only
    differences = torch.abs(v1_non_zero - v2_non_zero)
    close_scores = (differences < threshold).float()

    return close_scores.sum() / len(v1_non_zero)


def pattern_matching(v1, v2):
    """
    Calculates a weighted pattern matching score between two vectors. This function
    uses inverse differences as weights (small differences get higher weights) and
    applies a dynamic threshold based on the mean and standard deviation of the differences.
    The differences are scaled relative to the sum of absolute values of vector elements,
    with a small constant added to avoid division by zero.

    Parameters:
    v1 (torch.Tensor): First vector.
    v2 (torch.Tensor): Second vector.

    Returns:
    float: Weighted pattern similarity score (0 to 1 range), where 1 indicates perfect similarity.
    """
    differences = torch.abs(v1 - v2)
    # Inverse differences weights (smaller differences get higher weight)
    inverse_diff_weights = 1 / (differences + 0.01)  # +0.01 to avoid division by zero
    # Normalize weights to sum to 1
    importance_weights = inverse_diff_weights / inverse_diff_weights.sum()
    # Dynamic threshold based on a simple fixed factor
    threshold = differences.mean() + 0.1 * differences.std()
    # Calculate weighted differences using scaled difference
    scaled_differences = differences / (torch.abs(v1 + v2) + 0.1)  # +0.1 to avoid division by zero
    # Create a mask where scaled differences are less than the dynamic threshold
    close_scores = (scaled_differences < threshold).float()
    # Calculate the pattern similarity score
    pattern_similarity = (close_scores * importance_weights).sum() / importance_weights.sum()
    return pattern_similarity


def _get_loc_from_hook_name(hook_name):
    if "pre" in hook_name:
        return "pre"
    elif "post" in hook_name:
        return "post"
    elif "mid" in hook_name:
        return "mid"