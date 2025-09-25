import requests
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)

# Add CORS support to your Flask app
CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers="*")

BASE_URL = "https://aviationweather.gov/api/data"
USER_AGENT = "MyMetarApp/0.1"  # Simple valid User-Agent string
HEADERS = {'User-Agent': USER_AGENT}

# Helper function to query any endpoint with optional airport IDs
def fetch_weather_data(endpoint, airport_ids=None):
    params = {'format': 'json'}
    if airport_ids:
        params['ids'] = airport_ids  # filter by airports where applicable
    url = f"{BASE_URL}/{endpoint}"
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 204:
            return {"message": f"No data available for {endpoint}"}
        else:
            return {"error": f"API returned status {response.status_code} for {endpoint}"}
    except Exception as e:
        return {"error": f"Exception for {endpoint}: {str(e)}"}


@app.route('/weather', methods=['POST'])
def get_weather():
    data = request.get_json()
    airports = data.get('airports', [])
    if not airports or not isinstance(airports, list):
        return jsonify({"error": "Invalid input, please provide a list of airport ICAO codes."}), 400

    # Validate ICAO codes
    valid_airports = [a.upper() for a in airports if len(a) == 4 and a.isalnum()]
    if not valid_airports:
        return jsonify({"error": "No valid ICAO codes provided."}), 400

    airport_ids = ",".join(valid_airports)

    result = {
        "metar": fetch_weather_data("metar", airport_ids),
        "taf": fetch_weather_data("taf", airport_ids),
        "pirep": fetch_weather_data("pirep", airport_ids),
        "airsigmet": fetch_weather_data("airsigmet"),
        "isigmet": fetch_weather_data("isigmet"),
        "gairmet": fetch_weather_data("gairmet"),
        "airmet": fetch_weather_data("airmet"),
        "tcf": fetch_weather_data("tcf"),
        "cwa": fetch_weather_data("cwa"),
        "windtemp": fetch_weather_data("windtemp")
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
