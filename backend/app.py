import os, re, json, requests
import streamlit as st

# ----------------- GenAI client (DeepSeek) -----------------
DEEPSEEK_API_KEY = os.getenv("sk-28c7a916f31b4acdbf619226e86fd578")  # set this in your shell
DEEPSEEK_BASE = "https://api.deepseek.com/v1"     # OpenAI-compatible base [Docs]
DEEPSEEK_MODEL = "deepseek-chat"                  # non-thinking chat mode

def deepseek_brief_from_tokens(tokens: dict) -> str:
    if not DEEPSEEK_API_KEY:
        return ""
    prompt = (
        "Summarize these METAR tokens for pilots in 1–2 sentences. "
        "Lead with hazards/operational limits. Preserve numbers/units exactly. "
        "Omit fields not present.\n"
        f"Tokens: {json.dumps(tokens, ensure_ascii=False)}"
    )
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 160,
    }
    try:
        r = requests.post(f"{DEEPSEEK_BASE}/chat/completions", headers=headers, json=body, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""

# ----------------- Tokenizer -----------------
WX_CODES = r"(TS|DZ|RA|SN|SG|PL|GR|GS|UP|BR|FG|FU|HZ|DU|SA|SQ|PO|DS|SS|VA)"
CLOUD_CODES = r"(FEW|SCT|BKN|OVC)"

def tokenize_metar(raw: str) -> dict:
    t = {"station": None,"time": None,"wind": None,"vis": None,"wx": [],"clouds": [],"vv": None,"temp": None,"dew": None,"alt": None,"rvr": []}
    s = raw.strip()
    if s.startswith("METAR ") or s.startswith("SPECI "):
        s = s.split(" ", 1)[1]
    m = re.match(r"^([A-Z0-9]{4}) (\d{6})Z", s)
    if m: t["station"], t["time"] = m.group(1), m.group(2)
    m = re.search(r"\b(\d{3})(\d{2,3})(G\d{2,3})?KT\b", s)
    if m: t["wind"] = {"dir": m.group(1), "spd": m.group(2), "gst": (m.group(3) or "").lstrip("G")}
    m = re.search(r"\b(\d{1,2}(?:/\d{1,2})?SM)\b", s)
    if m: t["vis"] = m.group(1)
    t["wx"] = [("".join(g)).strip() for g in re.findall(r"\s(\+|-)?(VC)?"+WX_CODES+r"\b", s)]
    t["clouds"] = [(m.group(1), m.group(2)) for m in re.finditer(r"\b"+CLOUD_CODES+r"(\d{3})\b", s)]
    m = re.search(r"\bVV(\d{3})\b", s)
    if m: t["vv"] = int(m.group(1)) * 100
    m = re.search(r"\s(M?\d{2})/(M?\d{2})(\s|$)", s)
    if m:
        td = lambda x: ("-" + x[1:]) if x.startswith("M") else x
        t["temp"], t["dew"] = td(m.group(1)), td(m.group(2))
    m = re.search(r"\bA(\d{4})\b", s)
    if m:
        alt = m.group(1); t["alt"] = f"{alt[:2]}.{alt[2:]} inHg"
    t["rvr"] = re.findall(r"\bR(\d{2}[LRC]?)/([MP]?\d{3,4})(V([MP]?\d{3,4}))?FT\b", s)
    return t

# ----------------- Deterministic fallback -----------------
def deterministic_summary(t: dict) -> str:
    parts = []
    ceil = None
    if t.get("vv"): ceil = t["vv"]
    else:
        bases = [int(b)*100 for c,b in t.get("clouds", []) if c in ("BKN","OVC")]
        ceil = min(bases) if bases else None
    if (t.get("vis") in ["1/4SM","1/2SM","3/4SM","1SM","2SM"]) or (ceil is not None and ceil < 1000):
        parts.append("IFR conditions")
    if t.get("vis"): parts.append(f"Visibility {t['vis']}")
    if ceil: parts.append(f"Ceiling {ceil} ft")
    elif t.get("clouds"): parts.append("Clouds " + ", ".join([f"{c} {int(b)*100} ft" for c,b in t["clouds"]]))
    if t.get("wind"):
        w=t["wind"]; s=f"Winds {w['dir']}° at {w['spd']} kt"
        if w.get("gst"): s+=f", gusting {w['gst']} kt"
        parts.append(s)
    if t.get("wx"): parts.append("Weather " + ", ".join(t["wx"]))
    if t.get("temp") and t.get("dew"): parts.append(f"Temp {t['temp']}°C, dew {t['dew']}°C")
    if t.get("alt"): parts.append(f"Altimeter {t['alt']}")
    return ". ".join(parts) + "."

# ----------------- Fetch METAR from AWC -----------------
AWC_BASE = "https://aviationweather.gov/api/data"

def fetch_latest_metar(icao: str) -> str:
    try:
        r = requests.get(f"{AWC_BASE}/metar", params={"format":"json","ids":icao}, headers={"User-Agent":"WxBrief/0.1"}, timeout=12)
        r.raise_for_status()
        data = r.json()
        arr = data if isinstance(data, list) else data.get("metar", [])
        if not arr: return ""
        raw = arr[0].get("rawOb") or arr[0].get("raw_text") or ""
        return raw
    except Exception:
        return ""

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Aviation Weather Brief", page_icon="⛅", layout="centered")  # UI basics
st.title("Aviation Weather Brief (Quick)")                                                   # Streamlit basics

mode = st.radio("Input type", ["Airport ICAO", "Raw METAR"])
if mode == "Airport ICAO":
    icao = st.text_input("Enter ICAO (e.g., KJFK, VOBL)").upper().strip()
    if st.button("Fetch & Summarize"):
        if len(icao) == 4:
            raw = fetch_latest_metar(icao)
            if not raw:
                st.error("No METAR found right now.")
            else:
                tokens = tokenize_metar(raw)
                llm = deepseek_brief_from_tokens(tokens)
                summary = llm or deterministic_summary(tokens)
                st.code(raw, language="text")
                st.json(tokens)
                st.success(summary)
        else:
            st.warning("Please enter a 4‑letter ICAO.")
else:
    raw_in = st.text_area("Paste raw METAR", height=120, placeholder="METAR KJFK 121751Z 18016G24KT 4SM -RA BR BKN015 OVC025 06/04 A2990")
    if st.button("Summarize pasted METAR"):
        if raw_in.strip():
            tokens = tokenize_metar(raw_in)
            llm = deepseek_brief_from_tokens(tokens)
            summary = llm or deterministic_summary(tokens)
            st.code(raw_in.strip(), language="text")
            st.json(tokens)
            st.success(summary)
        else:
            st.warning("Paste a METAR first.")
