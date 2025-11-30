from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
from dotenv import load_dotenv
import openai

# 讀取 .env
load_dotenv()
CWA_API_KEY = os.getenv("CWA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 初始化 OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# 開啟 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可改成前端網址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/weather-summary")
def weather_summary(city: str = "花蓮縣"):
    api_url = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0001-001?Authorization={CWA_API_KEY}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        stations = data['records']['Station']
    except Exception as e:
        return {"error": f"無法取得氣象資料: {e}"}

    # 找指定縣市站點
    city_station = next(
        (s for s in stations if s['GeoInfo']['CountyName'] == city),
        None
    )
    if not city_station:
        return {"error": f"找不到城市 {city}"}

    try:
        weather_elem = city_station['WeatherElement']
        temp = float(weather_elem['AirTemperature'])
        humd = float(weather_elem['RelativeHumidity'])
        rain = float(weather_elem['Now']['Precipitation'])
        desc = weather_elem['Weather']
    except KeyError as e:
        return {"error": f"資料缺少欄位: {e}"}

    # LLM 生成溫柔描述
    prompt = f"請用溫和、友善的語氣描述台灣{city}目前的天氣。天氣：{desc}，溫度：{temp}°C，濕度：{humd}%，雨量：{rain}mm。"

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        summary = completion.choices[0].message.content
    except Exception as e:
        summary = f"無法取得溫柔描述: {e}"

    return {
        "city": city,
        "temperature": temp,
        "humidity": humd,
        "precipitation": rain,
        "description": desc,
        "summary": summary
    }
