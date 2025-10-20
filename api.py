import threading
import time
import argparse
import traceback
import asyncio
import logging

import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Response, status, HTTPException, Depends

from backend.cds_loader import CDSLoader
from backend.model import Predictor
from backend.prediction import Prediction
from backend.utils import slice_bbox

class AuroraAPI:
    PRED_SAFE_PATH = "~/.cache/aurora/prediction.npz"

    def __init__(
            self, 
            port,
            api_key: str,
            steps = 48, # 5 past + 7 future days, 12 * 4 = 48
        ):
        
        self.data_loader = CDSLoader(cache_dir="data/era5/")
        self.model = Predictor()
        self.steps = steps
        self.api_key = api_key

        self.prediction = Prediction.from_file(self.PRED_SAFE_PATH)
        self.prediction_needed = threading.Event()
        if self.data_loader.update_and_status() or self.prediction is None:
            self.prediction_needed.set()

        self.app = FastAPI()
        self.app.post('/query', dependencies=[Depends(self.verify)])(self.query)
        # run FastAPI in a separate thread.
        threading.Thread(target=lambda: uvicorn.run(self.app, host="0.0.0.0", port=port, log_level="info")).start()
        threading.Thread(target=self.timer_loop, daemon=True).start()
        self.worker() # block main thread for GPU

    def worker(self):
        while True:
            self.prediction_needed.wait()
            self.data_loader.update_and_status()
            logging.warning(f"{pd.Timestamp.now()} Running new prediction with steps {self.steps}...")
            self.prediction = self.run_prediction()
            self.prediction.to_file(self.PRED_SAFE_PATH)
            self.prediction_needed.clear()
           

    def timer_loop(self):
        while True:
            next_stamp = self.data_loader.last_six_hour_floored_stamp() + pd.Timedelta(hours=6)
            seconds = (pd.Timestamp.now() - next_stamp).total_seconds() + 1
            
            time.sleep(seconds)
            self.prediction_needed.set()

    
    def run_prediction(self) -> Prediction:
        self.data_loader.update_and_status()
        batch = self.model.construct_batch(
            static_vars=self.data_loader.static_vars,
            surf_vars_ds=self.data_loader.surf_vars_ds,
            atmos_vars_ds=self.data_loader.atmos_vars_ds
        )
        return self.model.predict(batch, self.steps)
    
    async def verify(self, request: Request):
        api_key = request.headers.get("Authorization", None)
        if api_key != self.api_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    async def query(self, request: Request):
        if self.prediction is None:
            return Response("Prediction is not ready yet.", status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        
        try:
            json = await request.json()
            lat_start = json["lat_start"]
            lat_end = json["lat_end"]
            
            lon_start = json["lon_start"]
            lon_end = json["lon_end"]
            start_time = pd.Timestamp(json["start_time"])
            end_time = pd.Timestamp(json["end_time"])

            variable = json["variable"]
            assert variable in self.prediction.__dict__, f"Only {self.prediction.__dict__.keys()} are valid variables."
            data = getattr(self.prediction, variable)

            # include both start and end time
            hours = int((end_time - start_time).total_seconds()) // 3600 + 1
            assert hours < 24

            # one hour more than data is first stored prediction
            start_idx = int((start_time - self.data_loader.surf_vars_ds.valid_time.max().values).total_seconds()) // 3600 - 1
            assert start_idx >= 0 and (start_idx + hours) < data.shape[0], "Start time is too early or late."

            result: np.ndarray = slice_bbox(
                data[start_idx:start_idx + hours],
                (lat_start, lat_end, lon_start, lon_end),
                lat_dim=1,
            )
                    
            return {"data": result.tolist(), "code": 200}

        except Exception:
            exc_string = traceback.format_exc()
            logging.warning("Error in request: \n" + exc_string)
            raise HTTPException(status_code=500, detail=exc_string)


# Create the API instance and run the server
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, required=True)
    parser.add_argument('-k', '--key', type=str, default="Weruletheatmosphere")
    args = parser.parse_args()

    api = AuroraAPI(port=args.port, api_key=args.key)