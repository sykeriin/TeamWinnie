# Install dependency with: pip install metar

from Metar import Metar

def metar_to_nlp(raw_metar: str) -> str:
    """
    Decodes a raw METAR string into a readable natural language weather briefing
    suitable for pilots, highlighting key weather elements concisely.
    """
    try:
        # Parse METAR code
        obs = Metar.Metar(raw_metar)

        # Build readable report parts
        report = []

        # Station and time
        if obs.station_id:
            report.append(f"Airport: {obs.station_id}")
        if obs.time:
            report.append(f"Observed at {obs.time.strftime('%H:%MZ on %Y-%m-%d')}")

        # Flight category for quick flight condition reference
        if obs.flight_category:
            report.append(f"Flight category: {obs.flight_category}")

        # Weather phenomena (e.g., rain, fog)
        if obs.weather:
            weather_desc = " and ".join(str(w) for w in obs.weather)
            report.append(f"Weather: {weather_desc}")

        # Temperature and dew point in Celsius
        if obs.temp:
            report.append(f"Temperature: {obs.temp.value()}°C")
        if obs.dewpt:
            report.append(f"Dew point: {obs.dewpt.value()}°C")

        # Wind details with direction (degrees) and speed (knots)
        if obs.wind_speed:
            wind = f"Winds from {obs.wind_dir.value()}° at {obs.wind_speed.value()} knots"
            if obs.wind_gust:
                wind += f", gusting to {obs.wind_gust.value()} knots"
            report.append(wind)

        # Horizontal visibility
        if obs.vis:
            report.append(f"Visibility: {obs.vis.value()} {obs.vis.unit}")

        # Cloud layers description
        clouds = [str(cloud) for cloud in obs.sky]
        if clouds:
            report.append(f"Cloud coverage: {', '.join(clouds)}")

        # Altimeter/Pressure reading
        if obs.press:
            report.append(f"Pressure: {obs.press.value()} {obs.press.unit}")

        # Join all components into a concise paragraph
        return ". ".join(report) + "."

    except Exception as e:
        # Handle any parsing errors gracefully
        return f"Error decoding METAR data: {e}"

def generate_pilot_briefing(airport_name: str, raw_metar: str) -> str:
    """
    Combine airport name with decoded weather to return a pilot-friendly briefing string.
    """
    briefing = metar_to_nlp(raw_metar)
    return f"{airport_name} Weather Briefing:\n{briefing}"

