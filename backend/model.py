from pathlib import Path

import torch
import numpy as np
import xarray as xr
from scipy.interpolate import PchipInterpolator

from aurora import Batch, Metadata, Aurora, rollout
import logging
from prediction import Prediction

class Predictor:

    def __init__(
            self, 
        ):
        self.aurora = Aurora(use_lora=False)  # "you can turn off LoRA to obtain more realistic predictions at the expensive of slightly higher long-term MSE"
        self.aurora = self.aurora.eval().to("cuda")
        print("Num Aurora parameters: ", sum([p.numel() for p in self.aurora.parameters()]))


    def construct_batch(self, static_vars: xr.Dataset, surf_vars_ds: xr.Dataset, atmos_vars_ds: xr.Dataset):
        """
        NOTE: function assumes that both datasets have only two timepoints. If more, it will ignore those.
        """
        return Batch(
            surf_vars={
                # inserts a batch dimension of size one.
                "2t": torch.from_numpy(surf_vars_ds["t2m"].values[:2][None]),
                "10u": torch.from_numpy(surf_vars_ds["u10"].values[:2][None]),
                "10v": torch.from_numpy(surf_vars_ds["v10"].values[:2][None]),
                "msl": torch.from_numpy(surf_vars_ds["msl"].values[:2][None]),
            },
            static_vars=static_vars,
            atmos_vars={
                "t": torch.from_numpy(atmos_vars_ds["t"].values[:2][None]),
                "u": torch.from_numpy(atmos_vars_ds["u"].values[:2][None]),
                "v": torch.from_numpy(atmos_vars_ds["v"].values[:2][None]),
                "q": torch.from_numpy(atmos_vars_ds["q"].values[:2][None]),
                "z": torch.from_numpy(atmos_vars_ds["z"].values[:2][None]),
            },
            metadata=Metadata(
                lat=torch.from_numpy(surf_vars_ds.latitude.values), # decreasing 90 to -90
                lon=torch.from_numpy(surf_vars_ds.longitude.values), # increasing 0 to 360
                # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
                # `datetime.datetime`s. Note that this needs to be a tuple of length one:
                # one value for every batch element. Select element 1,
                time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[1],),
                atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
            ),
        )
    
    def _post_process_generic(self, data, add_first_row=True):
        # aurora post-process
        # Make latitude go from -87.5 to 90.0 instead of decreasing order
        # duplicate the first row for a -90.0 prediction
        data = data.squeeze().flip(dims=(-2,))
        if add_first_row:
            data = torch.cat((data[-2].unsqueeze(0), data))
        return data
    
    def interpolate_wind_to_height(self, wind_data, geopotential_data, target_height):
        """
        MADE BY GEMINI - NOT TESTED
        Interpolates wind speed to a target geometric height using vectorized operations.
        Assumes input data is sorted by ascending geopotential along dimension 0.

        Args:
            wind_data (torch.Tensor): Wind speeds on pressure levels. Shape: (levels, lat, lon).
            geopotential_data (torch.Tensor): Geopotential on pressure levels, sorted low to high. Shape: (levels, lat, lon).
            target_height (float): The geometric height (m) to interpolate to.-

        Returns:
            torch.Tensor: Interpolated wind speed at the target height. Shape: (1, lat, lon).
        """
        g0 = 9.80665  # Standard gravity

        # Convert geopotential (m^2/s^2) to geometric height (m)
        # Assumes geopotential_data is already sorted from low to high.
        heights = geopotential_data / g0

        # Find the index of the level just below the target height for each grid point
        # The sum of a boolean mask gives the count of levels below the target.
        lower_indices = torch.sum(heights < target_height, dim=0) - 1
        lower_indices = lower_indices.clamp(min=0, max=heights.shape[0] - 2) # Clamp to valid range

        # Get indices for the level above
        upper_indices = lower_indices + 1

        # Gather the bracketing heights and wind speeds using the calculated indices
        h_lower = torch.gather(heights, 0, lower_indices.unsqueeze(0))
        h_upper = torch.gather(heights, 0, upper_indices.unsqueeze(0))
        w_lower = torch.gather(wind_data, 0, lower_indices.unsqueeze(0))
        w_upper = torch.gather(wind_data, 0, upper_indices.unsqueeze(0))

        # Calculate interpolation weight, avoiding division by zero
        epsilon = 1e-6
        weight = (target_height - h_lower) / (h_upper - h_lower + epsilon)

        # Perform the linear interpolation
        wind_at_target_height = w_lower + weight * (w_upper - w_lower)

        return wind_at_target_height

    def get_100m_wind(self, batch, var):
        return self.interpolate_wind_to_height(
            # if batch is more than one time step, take the last one
            batch.atmos_vars[var][:, -1].squeeze(),
            batch.atmos_vars["z"][:, -1].squeeze(),
            target_height=100.0
        )

    def predict(self, batch: Batch, steps) -> Prediction:
        # also include last of batch, so that interpolation for first 6 hours is possible
        temp_preds = [self._post_process_generic(batch.surf_vars["2t"][:, -1], add_first_row=False)]
        east_wind_preds = [self._post_process_generic(self.get_100m_wind(batch, "u"), add_first_row=False)]
        north_wind_preds = [self._post_process_generic(self.get_100m_wind(batch, "v"), add_first_row=False)]

        with torch.inference_mode():
            preds = [pred.to("cpu") for pred in rollout(self.aurora, batch, steps=steps)]

        for pred in preds:
            temp_preds.append(self._post_process_generic(pred.surf_vars["2t"]))
            east_wind_preds.append(self._post_process_generic(self.get_100m_wind(pred, "u")))
            north_wind_preds.append(self._post_process_generic(self.get_100m_wind(pred, "v")))     

        return Prediction(
            temperature = self.hermit_interp(torch.stack(temp_preds, dim=0).numpy()),
            u_wind_100m = self.hermit_interp(torch.stack(east_wind_preds, dim=0).numpy()),
            v_wind_100m = self.hermit_interp(torch.stack(north_wind_preds, dim=0).numpy())
        )   

    def hermit_interp(self, data: np.ndarray) -> np.ndarray:
        """
        Interpolates a tensor with 6-hourly timestamps to hourly resolution
        using Piecewise Cubic Hermite Interpolating Polynomial (PCHIP).

        Args:
            data_6hr (np.ndarray): The input data tensor with shape (x, 721, 1440).

        Returns:
            np.ndarray: The interpolated data tensor with shape (6(x-1), 721, 1440).
        """
        # Original times: 0, 6, 12, ..., 246 hours (41 * 6)
        t_original = np.arange(data.shape[0]) * 6

        # New times: 0, 1, 2, ..., 246 hours.
        t_new = np.arange(t_original[-1] + 1)

        interpolator = PchipInterpolator(t_original, data, axis=0)

        data_hourly = interpolator(t_new)
        # remove first datapoint (only there for interpolation)
        return data_hourly[1:]
