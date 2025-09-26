import re 


WX_CODES = r"(TS|DZ|RA|SN|SG|PL|GR|GS|UP|BR|FG|FU|HZ|DU|SA|SQ|PO|DS|SS|VA)"
CLOUD_CODES = r"(FEW|SCT|BKN|OVC)"

def tokenize_metar(raw: str) -> dict:
    t = {"station": None,"time": None,"wind": None,"vis": None,"wx": [],"clouds": [],"vv": None,"temp": None,"dew": None,"alt": None,"rvr": []}
    # Strip leading METAR/SPECI tag if present
    if raw.startswith("METAR ") or raw.startswith("SPECI "):
        raw = raw.split(" ", 1)[1]

    # Station and time
    m = re.match(r"^([A-Z0-9]{4}) (\d{6})Z", raw)
    if m:
        t["station"], t["time"] = m.group(1), m.group(2)

    # Wind dddff(Ggg)?KT
    m = re.search(r"\b(\d{3})(\d{2,3})(G\d{2,3})?KT\b", raw)
    if m:
        t["wind"] = {"dir": m.group(1), "spd": m.group(2), "gst": (m.group(3) or "").lstrip("G")}

    # Visibility nSM or n/nnSM
    m = re.search(r"\b(\d{1,2}(?:/\d{1,2})?SM)\b", raw)
    if m:
        t["vis"] = m.group(1)

    # Present weather codes (with optional intensity and vicinity)
    t["wx"] = [("".join(g)).strip() for g in re.findall(r"\s(\+|-)?(VC)?"+WX_CODES+r"\b", raw)]

    # Clouds FEW/SCT/BKN/OVCnnn
    t["clouds"] = re.findall(r"\b"+CLOUD_CODES+r"(\d{3})\b", raw)

    # Vertical visibility VV###
    m = re.search(r"\bVV(\d{3})\b", raw)
    if m:
        t["vv"] = int(m.group(1)) * 100

    # Temperature/dewpoint M##/## or ##/##
    m = re.search(r"\s(M?\d{2})/(M?\d{2})(\s|$)", raw)
    if m:
        td = lambda s: ("-" + s[1:]) if s.startswith("M") else s
        t["temp"], t["dew"] = td(m.group(1)), td(m.group(2))

    # Altimeter A####
    m = re.search(r"\bA(\d{4})\b", raw)
    if m:
        alt = m.group(1)
        t["alt"] = f"{alt[:2]}.{alt[2:]} inHg"

    # Runway visual range Rxx/####FT with optional variability
    t["rvr"] = re.findall(r"\bR(\d{2}[LRC]?)/([MP]?\d{3,4})(V([MP]?\d{3,4}))?FT\b", raw)

    return t

