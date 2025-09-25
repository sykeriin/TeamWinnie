import re
from datetime import datetime

def metar_to_nlp(raw_metar):
    """
    Decodes a raw METAR string into a brief, readable weather summary for pilots.
    """
    report = []

    # 1. Extract station code (ICAO)
    station = re.match(r'^(\w{4})', raw_metar)
    if station:
        report.append(f"Airport: {station.group(1)}")

    # 2. Extract observation time (DDHHMMZ)
    time_match = re.search(r'(\d{6})Z', raw_metar)
    if time_match:
        day = time_match.group(1)[:2]
        hour = time_match.group(1)[2:4]
        minute = time_match.group(1)[4:6]
        report.append(f"Observed at {hour}:{minute}Z on day {day}")

    # 3. Extract wind info (dddffKT or dddffGggKT)
    wind_match = re.search(r'(\d{3})(\d{2,3})(G\d{2,3})?KT', raw_metar)
    if wind_match:
        direction = wind_match.group(1)
        speed = wind_match

