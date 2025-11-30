# Streamlit CWA Weather Dashboard
# File: streamlit_cwa_weather_dashboard.py
# Requirements: streamlit, requests, pandas, openai, pydeck
# Install: pip install streamlit requests pandas openai pydeck

import os
import time
from datetime import datetime
import requests
import pandas as pd
import streamlit as st

# ----------------------
# Config / Constants
# ----------------------
CWA_API_KEY = os.getenv("CWA_API_KEY")  # set your 中央氣象署 API key in env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # set your OpenAI (or compatible) API key in env

# Endpoints (CWA open data)
ENDPOINT_STATION_HOURLY = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0001-001"  # hourly station observations
ENDPOINT_RAINFALL_10MIN = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0002-001"  # rainfall 10-min

# ----------------------
# Helpers
# ----------------------

def fetch_cwa(endpoint, params=None, api_key=None):
    """Fetch JSON from CWA Open Data API."""
    if api_key is None:
        api_key = CWA_API_KEY
    if not api_key:
        raise RuntimeError("Missing CWA API key. Set environment variable CWA_API_KEY or provide in app.")
    params = params or {}
    params.update({"Authorization": api_key, "format": "JSON"})
    resp = requests.get(endpoint, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def parse_hourly_observations(jsondata):
    """Parse O-A0001-001 structure into a DataFrame with columns: StationId, StationName, DateTime, Temp, RH, Latitude, Longitude"""
    recs = []
    records = jsondata.get("records", {}).get("location", [])
    for loc in records:
        station_id = loc.get("stationId") or loc.get("StationId") or loc.get("StationId")
        station_name = loc.get("locationName") or loc.get("name")
        lat = loc.get("lat") or loc.get("StationLatitude") or (loc.get("locationPosition") or {}).get("latitude")
        lon = loc.get("lon") or loc.get("StationLongitude") or (loc.get("locationPosition") or {}).get("longitude")
        obs_time = loc.get("time", {}).get("obsTime") or loc.get("time")
        elements = {e.get("elementName"): e.get("elementValue") for e in loc.get("weatherElement", [])}
        # Common element names: TEMP, HUMD (sometimes RelativeHumidity or RH)
        temp = elements.get("TEMP") or elements.get("AirTemperature") or elements.get("AirTemperature")
        rh = elements.get("HUMD") or elements.get("RH") or elements.get("RelativeHumidity")
        recs.append({
            "StationId": station_id,
            "StationName": station_name,
            "DateTime": obs_time,
            "Temperature": try_parse_float(temp),
            "Humidity": try_parse_float(rh),
            "Latitude": try_parse_float(lat),
            "Longitude": try_parse_float(lon),
        })
    return pd.DataFrame.from_records(recs)


def parse_rainfall(jsondata):
    """Parse rainfall dataset into DataFrame with StationId, DateTime, Precipitation"""
    recs = []
    records = jsondata.get("records", {}).get("location", [])
    for loc in records:
        station_id = loc.get("stationId") or loc.get("StationId")
        station_name = loc.get("locationName")
        lat = loc.get("lat") or loc.get("StationLatitude")
        lon = loc.get("lon") or loc.get("StationLongitude")
        elements = {e.get("elementName"): e.get("elementValue") for e in loc.get("weatherElement", [])}
        precip = elements.get("RAIN") or elements.get("Precipitation") or elements.get("RAIN_1H")
        obs_time = loc.get("time", {}).get("obsTime") or loc.get("time")
        recs.append({
            "StationId": station_id,
            "StationName": station_name,
            "DateTime": obs_time,
            "Precipitation": try_parse_float(precip),
            "Latitude": try_parse_float(lat),
            "Longitude": try_parse_float(lon),
        })
    return pd.DataFrame.from_records(recs)


def try_parse_float(x):
    try:
        if x is None:
            return None
        # some APIs return strings or objects
        return float(x)
    except Exception:
        return None


# ----------------------
# LLM integration
# ----------------------

def generate_natural_language_summary(stations_df, top_n=5, model_name=None):
    """Send a short summary to an LLM and return the text. Uses OpenAI-compatible API if OPENAI_API_KEY set.
    The prompt asks for a gentle, user-friendly summary of temperature, humidity, precipitation for Taiwan."""
    # Prepare a compact report of top N stations by population or by temperature extremes
    if stations_df is None or stations_df.empty:
        return "目前沒有可用的觀測資料。"

    # choose noteworthy items: hottest, coldest, highest humidity, highest precipitation
    hottest = stations_df.loc[stations_df['Temperature'].idxmax()] if 'Temperature' in stations_df.columns and stations_df['Temperature'].notnull().any() else None
    coldest = stations_df.loc[stations_df['Temperature'].idxmin()] if 'Temperature' in stations_df.columns and stations_df['Temperature'].notnull().any() else None
    wettest = stations_df.loc[stations_df['Precipitation'].idxmax()] if 'Precipitation' in stations_df.columns and stations_df['Precipitation'].notnull().any() else None

    prompt_lines = [
        "請用中文、語氣溫和且適合給一般使用者的方式，根據下面的即時觀測摘要，寫一段 2-4 句的天氣說明（含地點與關鍵數值），以及 1-2 條簡短建議。例如：外出時要不要帶傘、穿著建議等：",
        "\n---",
    ]
    if hottest is not None:
        prompt_lines.append(f"最高溫：{hottest.get('StationName')} {hottest.get('Temperature')}°C")
    if coldest is not None:
        prompt_lines.append(f"最低溫：{coldest.get('StationName')} {coldest.get('Temperature')}°C")
    if wettest is not None:
        prompt_lines.append(f"雨量（最近觀測）：{wettest.get('StationName')} {wettest.get('Precipitation')} mm")
    prompt = "\n".join(prompt_lines)

    # Use OpenAI-compatible API if key present
    if not OPENAI_API_KEY:
        # If no API key, return a template summary instead of calling LLM
        return (
            "樣板：目前北部溫度約 20°C，部分山區較涼；中南部則較溫暖，午後沿海與山區有局部陣雨。出門建議攜帶薄外套與雨具。"
        )

    import openai
    openai.api_key = OPENAI_API_KEY
    model_name = model_name or os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.6,
        )
        text = response['choices'][0]['message']['content'].strip()
        return text
    except Exception as e:
        return f"呼叫 LLM 失敗：{e}"


# ----------------------
# Streamlit UI
# ----------------------

def main():
    st.set_page_config(page_title="台灣即時天氣 Dashboard", layout="wide")

    st.title("🇹🇼 台灣即時天氣 Dashboard（示範：Temperature / Humidity / Precipitation）")

    with st.sidebar:
        st.header("設定")
        api_key_input = st.text_input("中央氣象署 API Key", value=CWA_API_KEY or "", type="password")
        openai_key_input = st.text_input("LLM API Key (OpenAI-compatible，可選)", value=OPENAI_API_KEY or "", type="password")
        if st.button("儲存到環境（本次執行）"):
            # Update runtime values (won't persist across sessions)
            global CWA_API_KEY, OPENAI_API_KEY
            CWA_API_KEY = api_key_input.strip() or None
            OPENAI_API_KEY = openai_key_input.strip() or None
            st.success("已儲存（僅在本次執行有效）。")

        st.markdown("---")
        st.write("資料更新頻率：建議每 5 分鐘或依需求調整。")
        st.write("資料來源：中央氣象署開放資料平台 (opendata.cwa.gov.tw)")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("地圖與測站摘要")
        refresh = st.button("立即更新")
        if 'last_fetch' not in st.session_state:
            st.session_state['last_fetch'] = None

        # Fetch data
        try:
            with st.spinner("抓取觀測資料..."):
                hourly_json = fetch_cwa(ENDPOINT_STATION_HOURLY)
                rainfall_json = fetch_cwa(ENDPOINT_RAINFALL_10MIN)
                df_hourly = parse_hourly_observations(hourly_json)
                df_rain = parse_rainfall(rainfall_json)

                # merge on StationName if StationId missing
                merged = pd.merge(df_hourly, df_rain[['StationId', 'Precipitation']], on='StationId', how='left')
                # fallback: merge on StationName
                if merged.empty:
                    merged = pd.merge(df_hourly, df_rain[['StationName', 'Precipitation']], on='StationName', how='left')

                st.session_state['stations_df'] = merged
                st.session_state['last_fetch'] = datetime.now()
        except Exception as e:
            st.error(f"擷取資料失敗：{e}")
            return

        df = st.session_state.get('stations_df')
        st.write(f"資料時間：{st.session_state.get('last_fetch')}")

        if df is not None and not df.empty:
            # show top 10 by temperature
            st.dataframe(df[['StationName', 'Temperature', 'Humidity', 'Precipitation']].sort_values(by='Temperature', ascending=False).head(15))
            # map (if lat/lon present)
            if df['Latitude'].notnull().any() and df['Longitude'].notnull().any():
                map_df = df.dropna(subset=['Latitude', 'Longitude'])
                map_df = map_df.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})
                st.map(map_df[['lat', 'lon']].drop_duplicates())

    with col2:
        st.subheader("LLM 簡短說明（自動產生）")
        summary = generate_natural_language_summary(st.session_state.get('stations_df'))
        st.info(summary)

        st.markdown("---")
        st.subheader("重點數值（快速看）")
        if df is not None and not df.empty:
            hottest = df.loc[df['Temperature'].idxmax()]
            coldest = df.loc[df['Temperature'].idxmin()]
            wettest = df.loc[df['Precipitation'].idxmax()] if 'Precipitation' in df.columns and df['Precipitation'].notnull().any() else None
            st.metric("最高溫", f"{hottest['Temperature']} °C", delta=None)
            st.metric("最低溫", f"{coldest['Temperature']} °C", delta=None)
            if wettest is not None:
                st.metric("最高雨量（近觀測）", f"{wettest['Precipitation']} mm", delta=None)

    st.sidebar.markdown("---")
    st.sidebar.write("範例部署：")
    st.sidebar.code("streamlit run streamlit_cwa_weather_dashboard.py")


if __name__ == '__main__':
    main()
