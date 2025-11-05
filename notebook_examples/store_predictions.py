from pathlib import Path
import pandas as pd
import os
import cdsapi
import torch
import xarray as xr 
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

from aurora import Batch, Metadata, Aurora, rollout

import sys



from dataclasses import dataclass
from typing import Optional
import os

import numpy as np

@dataclass
class Prediction(object):

    temperature: np.ndarray  # Temperature data at 2m height

    def to_file(self, filename):
        """
        Save the prediction data to a file as float16 to save space.
        """
        data_to_save = self.temperature.astype(np.float16)
        
        # Use np.savez_compressed for better compression
        np.savez_compressed(
            os.path.expanduser(filename),
            temperature=data_to_save,
        )
    
    def from_file(filename) -> Optional['Prediction']:
        """
        Load the prediction data from a file.
        """
        try:
            data = np.load(os.path.expanduser(filename))
            return Prediction(
                temperature=data['temperature'],
            )
        except FileNotFoundError:
            return None

    def get(self, variable: str) -> Optional[np.ndarray]:
        """
        Get a specific variable from the prediction.
        """
        if variable == "2m_temperature":
            return self.temperature
        raise NotImplementedError(f"Variable {variable} not implemented.")

from scipy.interpolate import PchipInterpolator



c = cdsapi.Client(sleep_max=10)

DOWNLOAD_PATH = Path("../data/era5")
DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)


def download_era5_day(variable: str, timestamp: pd.Timestamp, filename:str):
        """
        Make a request to Copernicus. 
        Can only request one variable at a time for now, as it will otherwise zip them
        """
        request = {
            "product_type": ["reanalysis"],
            "variable": [variable],
            "year": [str(timestamp.year)],
            "month": [str(timestamp.month).zfill(2)],
            "day": [str(timestamp.day).zfill(2)],
            "time": [
                "00:00",
                "01:00",
                "02:00",
                "03:00",
                "04:00",
                "05:00",
                "06:00",
                "07:00",
                "08:00",
                "09:00",
                "10:00",
                "11:00",
                "12:00",
                "13:00",
                "14:00",
                "15:00",
                "16:00",
                "17:00",
                "18:00",
                "19:00",
                "20:00",
                "21:00",
                "22:00",
                "23:00",
            ],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        c.retrieve(
            "reanalysis-era5-single-levels", request, target=filename
        )


def hermit_interp(data: np.ndarray, prefix: int, step_size=6) -> np.ndarray:
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

# NOTE: Works
def _post_process_generic(data, add_first_row=True):
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


def process_prediction(batch: Batch, preds) -> Prediction:
        # also include last of batch, so that interpolation for first 6 hours is possible
        temp_preds = [_post_process_generic(batch.surf_vars["2t"][:, i], add_first_row=False) for i in [0, 1]]

        for pred in preds:
            temp_preds.append(_post_process_generic(pred.surf_vars["2t"]))  

        # Calculate the entire hourly forecast
        full_forecast = hermit_interp(torch.stack(temp_preds, dim=0).numpy(), prefix=2)
        
        # Slice off only the last 24 hours (the ground truth day)
        gt_day_forecast = full_forecast[-24:]

        return Prediction(
            temperature = gt_day_forecast 
        )

def rmse(a,b):
      return np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).mean())     

def download_static():
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "geopotential",
                "land_sea_mask",
                "soil_type",
            ],
            "year": "2023", # doesn't matter, doesn't change
            "month": "01",
            "day": "01",
            "time": "00:00",
            "format": "netcdf",
        },
        str(DOWNLOAD_PATH / "static.nc"),
    )
    print("Static variables downloaded!")


def download_data(file_name: Path, data_source, vars, pressure_levels, time=["00:00", "06:00", "12:00", "18:00"],   day_start: pd.Timestamp = None):
    params = {
            "product_type": "reanalysis",
            "variable": vars,
            "year": str(day_start.year),
            "month": str(day_start.month).zfill(2),
            "day": str(day_start.day).zfill(2),
            "time": time,
            "format": "netcdf",
        }
    if pressure_levels:
        params["pressure_level"] = pressure_levels

    file_name.parent.mkdir(parents=True, exist_ok=True)
    c.retrieve(
        data_source,
        params,
        str(file_name),
    )


def get_prediction(start_time: pd.Timestamp, end_time: pd.Timestamp, model=None):
    """
    Get a prediction for a given start and end time.
    """
    DAY_START = start_time
    DAY_END = end_time

    print( f"Getting prediction from {start_time} for ground truth day {end_time}" )
    if not (DOWNLOAD_PATH / "static.nc").exists():
        download_static()

    #Download the surface-level variables.
    surface_path = DOWNLOAD_PATH / "surf_vars" / f"{DAY_START.strftime('%Y-%m-%d')}-surface-level.nc"
    print(surface_path)
    if not surface_path.exists():
        download_data(surface_path, "reanalysis-era5-single-levels", ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"], None, day_start=DAY_START)
        print("Surface-level variables downloaded!")

    # Download the atmospheric variables.
    atmos_path = DOWNLOAD_PATH / "atmos_vars" / f"{DAY_START.strftime('%Y-%m-%d')}-atmospheric.nc"
    if not atmos_path.exists():
        download_data(atmos_path, "reanalysis-era5-pressure-levels", 
                    ["temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity", "geopotential"], 
                    ["50", "100", "150", "200", "250", "300", "400", "500", "600", "700", "850", "925", "1000"], day_start=DAY_START)
        print("Atmospheric variables downloaded!")

    static_vars_ds = xr.open_dataset(DOWNLOAD_PATH / "static.nc", engine="netcdf4")
    surf_vars_ds = xr.open_dataset(surface_path, engine="netcdf4")
    atmos_vars_ds = xr.open_dataset(atmos_path, engine="netcdf4")

    batch = Batch(
        surf_vars={
            # First select the first two time points: 00:00 and 06:00. Afterwards, `[None]`
            # inserts a batch dimension of size one.
            "2t": torch.from_numpy(surf_vars_ds["t2m"].values[:2][None]),
            "10u": torch.from_numpy(surf_vars_ds["u10"].values[:2][None]),
            "10v": torch.from_numpy(surf_vars_ds["v10"].values[:2][None]),
            "msl": torch.from_numpy(surf_vars_ds["msl"].values[:2][None]),
        },
        static_vars={
            # The static variables are constant, so we just get them for the first time.
            "z": torch.from_numpy(static_vars_ds["z"].values[0]),
            "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
            "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
        },
        atmos_vars={
            "t": torch.from_numpy(atmos_vars_ds["t"].values[:2][None]),
            "u": torch.from_numpy(atmos_vars_ds["u"].values[:2][None]),
            "v": torch.from_numpy(atmos_vars_ds["v"].values[:2][None]),
            "q": torch.from_numpy(atmos_vars_ds["q"].values[:2][None]),
            "z": torch.from_numpy(atmos_vars_ds["z"].values[:2][None]),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_vars_ds.latitude.values),
            lon=torch.from_numpy(surf_vars_ds.longitude.values),
            # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
            # `datetime.datetime`s. Note that this needs to be a tuple of length one:
            # one value for every batch element. Select element 1, corresponding to time
            # 06:00.
            time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[1],),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
        ),
    )
    num_days = (DAY_END - DAY_START).days + 1
    total_time_points = num_days * 4  # 4 time points per day
    steps = total_time_points - 1
    print(f"Calculated {steps} prediction steps for {num_days} days.")
    with torch.inference_mode():
        preds = [pred.to("cpu") for pred in rollout(model, batch, steps=steps)]
    print(len(preds))

    # DAY_END is ground truth day
    gt_day_str = DAY_END.strftime('%Y-%m-%d')
    # Calculate the lead time in days
    lead_time_days = (DAY_END - DAY_START).days

    pred_dir = DOWNLOAD_PATH / "predictions" / f"GT_{gt_day_str}"
    pred_dir.mkdir(parents=True, exist_ok=True)

    pred_safe_path = pred_dir / f"LEAD_{lead_time_days}d.npz" 
    temperature = process_prediction(batch, preds)
    temperature.to_file(pred_safe_path)




if __name__ == "__main__":

    # for 7 ground truth days, we want to get predictions from 5 days before until day occurs
    # let's say 7,8,9,10,11,12,13 october
    # for 7 october, we want to get predictions from 2 october until 7 october
    gt_day_start = pd.Timestamp("2025-10-7")
    gt_day_end = pd.Timestamp("2025-10-13")

    model = Aurora(use_lora=False)  # The pretrained version does not use LoRA.
    model.load_checkpoint(name="aurora-0.25-pretrained.ckpt")

    model.eval()
    model = model.to("cuda")
    print("Num parameters: ", sum([p.numel() for p in model.parameters()]))

    for gt_day in pd.date_range(gt_day_start, gt_day_end):
        print("ground truth day", gt_day)
        for days_before in range(1, 3):
            print(f"days_before: {days_before}")
            day_before = gt_day - pd.Timedelta(days=days_before)
            print('getting prediction for', day_before, day_before + pd.Timedelta(days=1))
            print(get_prediction(day_before, gt_day, model))
        #get_prediction(day, day + pd.Timedelta(days=1))