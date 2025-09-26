# weather_fetcher.py

import requests

def fetch_weather_data(endpoint, airport_ids=None):
    BASE_URL = "https://aviationweather.gov/api/data"
    USER_AGENT = "MyMetarApp/0.1"
    HEADERS = {'User-Agent': USER_AGENT}

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
