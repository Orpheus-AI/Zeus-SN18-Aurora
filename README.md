<p align="center">
  <a href="https://www.zeussubnet.com/">
  <img src="https://github.com/Orpheus-AI/Zeus/blob/main/static/zeus-icon.png" alt="Zeus Logo" width="150"/>
</a>
</p>

# Zeus (SN18) Aurora API

We provide an implementation of Aurora as FastAPI that fetches the latest ERA5 data and generates global forecasts to provide hourly 2m temperature predictions. This repository contains everything required to download the inputs from Copernicus, run the inference on a GPU, cache Aurora's predictions and serve them to downstream applications.

## Information about Aurora

- Aurora is a foundational weather model developed by Microsoft ([To read more](https://microsoft.github.io/aurora/intro.html)).
- The model is initialized with the latest ERA5 data (surface and atmospheric variables reflecting the current state of the atmosphere) to predict the following X steps of the atmosphere.
- Each prediction step is 6 hours. Therefore, we interpolate between two adjacent predictions to get hourly predictions.
- We use the aurora-0.25-pretrained.ckpt for ERA5.
- We set the prediction rollout to 49 steps: 5 past and 7 future days, 12 * 4 = 48 and 1 extra for hermite interpolation

## Repository Layout
- `api.py` – contains the FastAPI entrypoint, background refresh loop and request handler
- `backend/cds_loader.py` – ERA5 downloader and data loader
- `backend/model.py` – Aurora model wrapper, batch construction and hourly interpolation logic
- `backend/utils.py` – Bounding-box slicing shared by the API and notebooks
- `notebook_examples/test_api.ipynb` – api usage example

## Installation

### Prerequisites
- GPU with at least 48 GB VRAM
- Linux with Python 3.11+ 
- Copernicus Data Store (CDS) account **with API key** (setup instructions in [Zeus/validating.md](https://github.com/Orpheus-AI/Zeus/blob/main/docs/Validating.md#2-ecmwf))

### Steps
1. `cd /root/Zeus-SN18-Aurora`
2. Run `bash install.sh`
3. Update `settings.env` with your CDS credentials (`CDS_API_KEY=<your key here>`)
4. Run `bash install.sh` again to finish dependency installation and CDS login
5. Start the API with pm2:
   ```bash
   pm2 start /root/Zeus-SN18-Aurora/api.py --name "aurora-api" --interpreter python3 --no-autorestart -- -p 17200 
   ```

> The first boot performs the ERA5 downloads and the Aurora prediction rollout steps. This preparation step takes 10-15 min the first time. 

### settings.env
- `CDS_API_KEY` – Copernicus API token.

The installer writes the key to `~/.cdsapirc` so `cdsapi` can authenticate. Re-run `install.sh` whenever the key changes.

## Running the API

### ERA5 refresh
- `CDSLoader.update_and_status()` checks for new ERA5 data every 6 hours (00:00, 06:00, 12:00 and 18:00).
- When new data is detected, `Predictor.run_prediction()` constructs a batch with the latest surface and atmospheric variables.
- Predictions are interpolated to hourly resolution, saved to `~/.cache/aurora/prediction_steps49.npz`, and swapped automatically.
- A background timer triggers refreshes without interrupting the running API.

## API Reference

### POST `/query`
- **Headers:** `Authorization: <api_key>` (defaults to `Weruletheatmosphere`)
- **Body:**
```json
{
  "lat_start": 43,
  "lat_end": 44,
  "lon_start": -79,
  "lon_end": -78,
  "start_time": "2025-10-19T13:00:00",
  "end_time": "2025-10-19T16:00:00",
  "variable": "2m_temperature"
}
```

#### Parameters
We predict a bounding box (i.e. a slice of the earth)

- `lat_start`, `lat_end` – Latitude bounds in degrees (inclusive, -90 to 90)
- `lon_start`, `lon_end` – Longitude bounds in degrees (inclusive, -180 to 180)
- `start_time`, `end_time` – ISO timestamps with hourly resolution within the forecast window
- `variable` – Currently only supports `2m_temperature`

#### Responses
- `200 OK` – Returns `{ "data": [[[...]]], "code": 200 }` with shape `[hours, lat, lon]`
- `401 Unauthorized` – Missing or invalid API key
- `503 Service Unavailable` – Predictions still warming up (initial download or refresh in progress)
- `500 Internal Server Error` – Validation failure or unexpected exception 

> The service limits slices to 24 sequential hours per request and aligns time indices to the latest ERA5 assimilation. Align query windows with the `valid_time` of the dataset to avoid out-of-range errors.

## Example: `notebook_examples/test_api.ipynb`
- The notebook shows how to:
  - Construct a query payload with python
  - Authenticating with the API
  - Comparing Aurora output against Open-Meteo data

## Troubleshooting
- `503 Prediction is not ready yet` – Wait for the initial download/prediction cycle to finish.
- `Unauthorized` responses – Ensure the `Authorization` header matches the key.

## Credits

This implementation is built upon **Aurora: A Foundation Model for the Earth System**.

The original research and model are detailed in the following paper:
* **Docs:** [Aurora docs](https://microsoft.github.io/aurora/intro.html)
* **Paper:** [A Foundation Model for the Earth System](https://doi.org/10.1038/s41586-025-09005-y)
* **Authors:** Cristian Bodnar, Wessel P. Bruinsma, Ana Lucic, and colleagues.

If you plan to use this project or the model for **commercial purposes**, please contact the original developers at AIWeatherClimate@microsoft.com, as instructed in their documentation.

