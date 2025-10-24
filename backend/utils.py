from typing import Union, Tuple

import numpy as np
import torch

def slice_bbox(
    matrix: Union[np.ndarray, torch.Tensor], bbox: Tuple[float, float, float, float], lat_dim:int = 0,
) -> Union[np.ndarray, torch.Tensor]:
    """
    --- Taken from Zeus codebase, to reduce import dependencies ---
    Slice the matrix to the given lat-lon bounding box. This assumes that the matrix is of shape (180 * fidelity + 1, 360 * fidelity, ...).
    NOTE: it is also assumed that coordinates are in the range of -90 to 90 for latitude and -180 to 179.75 for longitude.

    Lat_dim can optionally be used to specify the dimension of the latitude data (defaults to 0)
     longitude dimension is assumed to be lat_dim + 1.
    """

    fidelity = matrix.shape[lat_dim + 1] // 360

    lat_start, lat_end, lon_start, lon_end = bbox
    lat_start_idx = int((90 + lat_start) * fidelity)
    lat_end_idx = int((90 + lat_end) * fidelity)
    lon_start_idx = int((180 + lon_start) * fidelity)
    lon_end_idx = int((180 + lon_end) * fidelity)

    # slice across specified dimensions only
    sl = [slice(None)] * matrix.ndim
    sl[lat_dim] = slice(lat_start_idx, lat_end_idx + 1)
    sl[lat_dim + 1] = slice(lon_start_idx, lon_end_idx + 1)

    return matrix[tuple(sl)]

def get_grid(
    lat_start: float,
    lat_end: float,
    lon_start: float,
    lon_end: float,
    fidelity: int = 4,
) -> torch.Tensor:
    """
    Get a grid of lat-lon points in the given bounding box.
    """
    return torch.stack(
        torch.meshgrid(
            *[
                torch.linspace(start, end, int((end - start) * fidelity) + 1)
                for start, end in [(lat_start, lat_end), (lon_start, lon_end)]
            ],
            indexing="ij",
        ),
        dim=-1,
    )  # (lat, lon, 2)