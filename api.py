import threading
import time
import argparse
import traceback
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO)

import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Response, status, HTTPException, Depends

from backend.cds_loader import CDSLoader
from backend.model import Predictor
from backend.prediction import Prediction
from backend.utils import slice_bbox

class AuroraAPI:

    def __init__(
            self, 
            port,
            api_key: str,
            steps = 49, # 5 past + 7 future days, 12 * 4 = 48 and 1 extra for hermite interpolation
        ):
        
        self.data_loader = CDSLoader(cache_dir="data/era5/")
        self.model = Predictor()
        self.steps = steps
        self.api_key = api_key

        self.pred_safe_path = Path.home() / ".cache" / "aurora" / f"prediction_steps{steps}.npz"
        self.pred_safe_path.parent.mkdir(parents=True, exist_ok=True)

        self.prediction = Prediction.from_file(self.pred_safe_path)
        self.prediction_needed = threading.Event()

        if self.data_loader.update_and_status() or self.prediction is None:
            self.prediction_needed.set()

        self.app = FastAPI()
        self.app.post('/query', dependencies=[Depends(self.verify)])(self.query)
        self.app.get('/refresh', dependencies=[Depends(self.verify)])(self.refresh)
        # run FastAPI in a separate thread.
        threading.Thread(target=lambda: uvicorn.run(self.app, host="0.0.0.0", port=port, log_level="info")).start()
        threading.Thread(target=self.timer_loop, daemon=True).start()
        self.worker() # block main thread for GPU

    def worker(self):
        while True:
            self.prediction_needed.wait()
            self.data_loader.update_and_status()

            logging.warning(f"[{pd.Timestamp.now()}] Running new prediction with steps {self.steps}...")
            self.prediction = self.run_prediction()
            self.prediction.to_file(self.pred_safe_path)
            logging.warning(f"[{pd.Timestamp.now()}] Prediction done.")

            self.prediction_needed.clear()
           

    def timer_loop(self):
        while True:
            now = pd.Timestamp.now(tz="UTC")
            next_stamp = (now + pd.Timedelta(seconds=1)).ceil("6h")
            seconds = (next_stamp - now).total_seconds()
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
            data = self.prediction.get(variable)

            assert start_time <= end_time, "start_time must be before or equal to end_time."

            # include both start and end time
            hours = int((end_time - start_time).total_seconds()) // 3600 + 1
            assert hours < 24

            # one hour more than data is first stored prediction
            start_idx = int((start_time - self.data_loader.surf_vars_ds.valid_time.max().values).total_seconds()) // 3600 - 1
            assert start_idx >= 0 and (start_idx + hours) < data.shape[0], "Start or end time is too early or late."

            result: np.ndarray = slice_bbox(
                data[start_idx:start_idx + hours],
                (lat_start, lat_end, lon_start, lon_end),
                lat_dim=1,
            )
                    
            return {"data": result.tolist(), "code": 200}

        except Exception:
            exc_string = traceback.format_exc()
            raise HTTPException(status_code=500, detail=exc_string)

    async def refresh(self):
        """
        Trigger an immediate data check and prediction refresh.
        """
        self.prediction_needed.set()
        return {"status": "queued"}


# Create the API instance and run the server
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, required=True)
    parser.add_argument('-k', '--key', type=str, default="Weruletheatmosphere")
    args = parser.parse_args()

    api = AuroraAPI(port=args.port, api_key=args.key)