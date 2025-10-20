if [ -f "settings.env" ]; then
    echo "File 'settings.env' already exists. We can proceed!"
else
    cat > settings.env << 'EOL'
# Enter your CDS API KEY for Aurora input data
CDS_API_KEY=
EOL
    echo "File 'settings.env' created. FILL IN THE REQUIRED SETTINGS BEFORE PROCEEDING!"
    exit 1
fi

set -a
source settings.env
set +a

if [ -z "$CDS_API_KEY" ]; then
    echo "Please specify a CDS API KEY to login to CDS! You will not be able to download live ERA5 data."
    exit 1
fi

rm $HOME/.cdsapirc
echo "url: https://cds.climate.copernicus.eu/api" > $HOME/.cdsapirc
echo "key: $CDS_API_KEY" >> $HOME/.cdsapirc

apt update -y
apt install -y \
    python3-pip \
    nano \
    npm
npm install -g pm2@latest

pip install -r requirements.txt