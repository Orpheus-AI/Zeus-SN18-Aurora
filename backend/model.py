from pathlib import Path

import torch
import numpy as np
import xarray as xr
from scipy.interpolate import PchipInterpolator

from aurora import Batch, Metadata, Aurora, rollout
import logging
from backend.prediction import Prediction

class Predictor:

    def __init__(
            self, 
        ):
        # according to Aurora docs: "you can turn off LoRA to obtain more realistic predictions at the expensive of slightly higher long-term MSE"
        self.aurora = Aurora(use_lora=False) 
        self.aurora.load_checkpoint(name="aurora-0.25-pretrained.ckpt")
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
            data = torch.cat((data[0].unsqueeze(0), data))
        # convert longitude 0..360 -> -180..180 by rolling half the width
        shift = data.shape[-1] // 2
        data = torch.roll(data, shifts=shift, dims=-1)
        return data

    def predict(self, batch: Batch, steps: int) -> Prediction:
        # also include batch itself, so that interpolation for first 6 hours is possible
        temp_preds = [self._post_process_generic(batch.surf_vars["2t"][:, i], add_first_row=False) for i in [0, 1]]

        with torch.inference_mode():
            for pred in rollout(self.aurora, batch, steps=steps):
                temp_preds.append(self._post_process_generic(pred.surf_vars["2t"].to("cpu")))

        return Prediction(
            temperature = self.hermit_interp(torch.stack(temp_preds, dim=0).numpy(), prefix=2),
        )   

    def hermit_interp(self, data: np.ndarray, prefix: int, step_size=6) -> np.ndarray:
        """
        Interpolates a tensor with 6-hourly timestamps to hourly resolution
        using Piecewise Cubic Hermite Interpolating Polynomial (PCHIP).
        # NOTE: last hour of data is only for interpolation purposes and truncated

        Args:
            data (np.ndarray): The input data tensor with shape (x, 721, 1440).
            prefix (int): Number of initial hours to skip in the output.

        Returns:
            np.ndarray: The interpolated data tensor with shape (step_size * (x-prefix) - 1, 721, 1440).
        """
        # Original times: 0, 6, 12, ..., 246 hours (41 * 6)
        t_original = np.arange(data.shape[0]) * step_size

        # New times: 0, 1, 2, ..., 246 hours.
        t_new = np.arange(t_original[-1] + 1)

        interpolator = PchipInterpolator(t_original, data, axis=0)

        data_hourly = interpolator(t_new)
        # remove first (couple) and last datapoint (only there for interpolation)
        start_preds = (prefix - 1) * step_size + 1
        return data_hourly[start_preds:-1]