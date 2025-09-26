from tokenizer import tokenize_metar  # in same folder
from gemini_client import gemini_brief_from_tokens

import requests
from flask import Flask, request, jsonify
#<<<<<<< HEAD
from flask_cors import CORS 
#=======
from flask_cors import CORS
#>>>>>>> 4c8e4522b00ea3a76df3d56d998fba157d55a1ee

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers="*")

BASE_URL = "https://aviationweather.gov/api/data"
USER_AGENT = "MyMetarApp/0.1"
HEADERS = {'User-Agent': USER_AGENT}

def fetch_weather_data(endpoint, airport_ids=None):
    params = {'format': 'json'}
    if airport_ids:
        params['ids'] = airport_ids
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

def simplify_airsigmet_isigmet(raw_data):
    simplified = []
    if not raw_data or "features" not in raw_data:
        return simplified
    for feature in raw_data["features"]:
        props = feature.get("properties", {})
        advisory = {
            "info": props.get("info"),
            "issueTime": props.get("issueTime"),
            "phenomenon": props.get("phenomenon"),
            "hazardType": props.get("hazardType"),
            "hazardSeverity": props.get("hazardSeverity"),
            "startTime": props.get("startTime"),
            "endTime": props.get("endTime"),
            "area": props.get("areaDescription"),
        }
        cleaned = {k: v for k, v in advisory.items() if v is not None}
        simplified.append(cleaned)
    return simplified

@app.route('/weather', methods=['POST'])
def get_weather():
    data = request.get_json()
    airports = data.get('airports', [])
    if not airports or not isinstance(airports, list):
        return jsonify({"error": "Invalid input, please provide a list of airport ICAO codes."}), 400

    valid_airports = [a.upper() for a in airports if len(a) == 4 and a.isalnum()]
    if not valid_airports:
        return jsonify({"error": "No valid ICAO codes provided."}), 400

    airport_ids = ",".join(valid_airports)

    metar = fetch_weather_data("metar", airport_ids)
    taf = fetch_weather_data("taf", airport_ids)
    pirep = fetch_weather_data("pirep", airport_ids)
    airsigmet_raw = fetch_weather_data("airsigmet")
    isigmet_raw = fetch_weather_data("isigmet")

    airsigmet = simplify_airsigmet_isigmet(airsigmet_raw)
    isigmet = simplify_airsigmet_isigmet(isigmet_raw)

    result = {
        "metar": metar,
        "taf": taf,
        "pirep": pirep,
        "airsigmet": airsigmet,
        "isigmet": isigmet,
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)