## 🚀 Installation


### Prerequisites

- GPU >= 48 RAM
- A CDS account with API key (Explained here [Zeus/validating.md](https://github.com/Orpheus-AI/Zeus/blob/main/docs/Validating.md#2-ecmwf))

### Steps

Follow these steps to set up the project locally:
1. cd into SN18-Aurora
2. Run ```bash install.sh```
3. Set CDS api key in settings.env
4. Run install again ```bash install.sh```
5. ```pm2 start api.py --name "api" --interpreter python3 --no-autorestart -- -p 17200 ```

It will take a while before the data is downloaded and the predictions are completed. 

### api.py

The API fetches the latest available ERA5 data and uses it to generate predictions. 
These predictions are cached such that it can be queried with the \query endpoint.
The data is refreshed every 6 hours with the latest ERA5 data to generate new predictions.  
