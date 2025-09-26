import requests
import json

# Test the health endpoint
def test_health():
    try:
        response = requests.get('http://localhost:5000/health')
        print("Health Check:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        print("-" * 50)
    except Exception as e:
        print(f"Health check failed: {e}")

# Test weather analysis
def test_weather_analysis():
    try:
        url = 'http://localhost:5000/weather/analysis'
        data = {
            "stations": ["KJFK", "KLAX"],
            "analysis_type": "severity"
        }
        
        response = requests.post(url, json=data)
        print("Weather Analysis:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        print("-" * 50)
    except Exception as e:
        print(f"Weather analysis failed: {e}")

# Test route weather
def test_route_weather():
    try:
        url = 'http://localhost:5000/weather/route'
        data = {
            "origin": "VOBL",
            "destination": "VEGT"
        }
        
        response = requests.post(url, json=data)
        print("Route Weather:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        print("-" * 50)
    except Exception as e:
        print(f"Route weather failed: {e}")

if __name__ == "_main_":
    print("Testing Aviation Weather API...")
    print("=" * 50)
    
    test_health()
    test_weather_analysis()
    test_route_weather()
    
    print("Testing complete!")