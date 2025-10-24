from pathlib import Path
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
        hours_age = (self.last_six_hour_floored_stamp() - self.surf_vars_ds.valid_time.max().values).total_seconds() / 3600
        if hours_age == 6:
            print("We are 6 hours behind. Updating data.")
            os.remove(self.surf_root / "0.nc")
            os.remove(self.atmos_root / "0.nc")
            os.rename(self.surf_root / "1.nc", self.surf_root / "0.nc")
            os.rename(self.atmos_root / "1.nc", self.atmos_root / "0.nc")
            self.download_date(
                self.last_six_hour_floored_stamp(), 
                self.surf_root / "1.nc", 
                self.atmos_root / "1.nc"
            )
        elif hours_age > 6:
            print(f"Data is too old. Re-downloading the two most recent days.")
            self.download_two_most_recent(self.surf_root, self.atmos_root)
        else:
            return False
        self.load_dataset()
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
        last_era5 = pd.Timestamp.now().floor("h") - pd.Timedelta(days=5)
        return last_era5 - pd.Timedelta(hours=last_era5.hour % 6)

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