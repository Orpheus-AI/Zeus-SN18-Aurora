from pathlib import Path
import math
import os

import pandas as pd
import cdsapi
import torch
import xarray as xr


class CDSLoader():

    def __init__(
            self,
            cache_dir: str, 
        ):
        self.cds_api = cdsapi.Client(sleep_max=10)

        self.cache_folder = Path(cache_dir).expanduser()
        self.cache_folder.mkdir(parents=True, exist_ok=True)

        static_path = self.cache_folder / "static.nc"
        self.surf_root = self.cache_folder / "surf_vars"
        os.makedirs(self.surf_root, exist_ok=True)
        self.atmos_root = self.cache_folder / "atmos_vars"
        os.makedirs(self.atmos_root, exist_ok=True)

        if not static_path.exists():
            self.download_static(static_path)
        self.static_vars = self.load_static(static_path)

        try:
            self.load_dataset()
        except:
            self.download_two_most_recent(self.surf_root, self.atmos_root)
            self.load_dataset()

    def load_static(self, path):
        static_vars_ds = xr.open_dataset(path, engine="netcdf4")
        return {
                # The static variables are constant, so we just get them for the first time.
                "z": torch.from_numpy(static_vars_ds["z"].values[0]),
                "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
                "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
            }


    def update_and_status(self) -> bool:
        latest_local = pd.Timestamp(self.surf_vars_ds.valid_time.max().values)
        target_time = self.last_six_hour_floored_stamp()
        delta = target_time - latest_local

        if delta <= pd.Timedelta(0):
            return False

        six_hours = pd.Timedelta(hours=6)
        cycles_behind = max(1, math.ceil(delta / six_hours))
        lag_hours = float(delta / pd.Timedelta(hours=1))

        if cycles_behind >= 2:
            print(f"Data is {lag_hours:.1f} hours behind. Re-downloading the two most recent steps.")
        else:
            print(f"Data is {lag_hours:.1f} hours behind. Refreshing the most recent step.")

        previous_latest = latest_local
        self.download_two_most_recent(self.surf_root, self.atmos_root)
        self.load_dataset()

        new_latest = pd.Timestamp(self.surf_vars_ds.valid_time.max().values)
        if new_latest <= previous_latest:
            print(f"No new ERA5 data available yet (latest timestamp still {previous_latest}).")
            return False

        return True
        
    def load_sharded_dataset(self, root: Path):
        dataset = xr.open_mfdataset(
            [root / "0.nc", root / "1.nc"], 
            combine="by_coords", 
            engine='netcdf4',
            compat="no_conflicts",
        )
        dataset = dataset.sortby("valid_time")
        return dataset
    
    def load_dataset(self):
        self.surf_vars_ds = self.load_sharded_dataset(self.surf_root)
        self.atmos_vars_ds = self.load_sharded_dataset(self.atmos_root)

    def download_static(self, path):
        print("Downloading static data (one-off)!")
        self.cds_api.retrieve(
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
            str(path),
        )


    def download_data(self, 
                      file_name, 
                      data_source, 
                      vars, 
                      pressure_levels,
                      day,
    ):
        params = {
                "product_type": "reanalysis",
                "variable": vars,
                "year": str(day.year),
                "month": str(day.month).zfill(2),
                "day": str(day.day).zfill(2),
                "time": str(day.hour).zfill(2) + ":00",
                "format": "netcdf",
            }
        if pressure_levels:
            params["pressure_level"] = pressure_levels

        self.cds_api.retrieve(
            data_source,
            params,
            str(file_name),
        )

    def last_six_hour_floored_stamp(self):
        now_utc = pd.Timestamp.now(tz="UTC").floor("h")
        last_era5 = now_utc - pd.Timedelta(days=5)
        base_utc = last_era5 - pd.Timedelta(hours=last_era5.hour % 6)  # 00/06/12/18 UTC
        return base_utc.tz_localize(None)

    def download_date(self, date, surf_file, atmos_file):
        print(f"Downloading data for {date}")
        # Surface variables
        self.download_data(
            surf_file,
            "reanalysis-era5-single-levels",
            [
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "mean_sea_level_pressure",
                "surface_pressure",
            ],
            None,
            date,
        )

        # Atmospheric variables
        self.download_data(
            atmos_file,
            "reanalysis-era5-pressure-levels",
            [
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
                "geopotential",
                "specific_humidity",
            ],
            ["50", "100", "150", "200", "250", "300", "400", "500", "600", "700", "850", "925", "1000"],
            date,
        )


    def download_two_most_recent(self, surf_root, atmos_root):
        """
        Download the two most recent days of surface and atmospheric data.
        """
        last_available = self.last_six_hour_floored_stamp()
        prev_step = last_available - pd.Timedelta(hours=6)

        for i, date in enumerate([prev_step, last_available]):
            self.download_date(date, surf_root / f"{i}.nc", atmos_root / f"{i}.nc")