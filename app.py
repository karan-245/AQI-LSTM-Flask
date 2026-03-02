from flask import Flask, render_template, request
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from datetime import date, timedelta

app = Flask(__name__)

# ---------------- AQI CATEGORY ---------------- #
def aqi_category(aqi):
    if aqi <= 50:
        return "Good 🟢"
    elif aqi <= 100:
        return "Satisfactory 🟡"
    elif aqi <= 200:
        return "Moderate 🟠"
    elif aqi <= 300:
        return "Poor 🔴"
    elif aqi <= 400:
        return "Very Poor 🟣"
    else:
        return "Severe ⚫"


aqi_values_table = [
    ("0–50", "Good 🟢", "Minimal impact"),
    ("51–100", "Satisfactory 🟡", "Minor discomfort to sensitive people"),
    ("101–200", "Moderate 🟠", "Breathing discomfort to vulnerable groups"),
    ("201–300", "Poor 🔴", "Breathing discomfort to most people"),
    ("301–400", "Very Poor 🟣", "Respiratory illness on prolonged exposure"),
    ("401–500", "Severe ⚫", "Serious health effects"),
]

# ---------------- API FUNCTIONS ---------------- #
def get_lat_lon(city):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    res = requests.get(url, params={"name": city, "count": 1}).json()
    if "results" not in res:
        return None, None
    return res["results"][0]["latitude"], res["results"][0]["longitude"]


def fetch_pm25(lat, lon):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5",
        "past_days": 365
    }

    res = requests.get(url, params=params).json()

    df = pd.DataFrame({
        "time": pd.to_datetime(res["hourly"]["time"]),
        "pm25": res["hourly"]["pm2_5"]
    })

    df = df.set_index("time").resample("D").mean().reset_index()
    return df


def pm25_to_aqi(pm):
    if pm <= 12:
        return (50 / 12) * pm
    elif pm <= 35.4:
        return ((100 - 51) / (35.4 - 12)) * (pm - 12) + 51
    elif pm <= 55.4:
        return ((150 - 101) / (55.4 - 35.4)) * (pm - 35.4) + 101
    elif pm <= 150.4:
        return ((200 - 151) / (150.4 - 55.4)) * (pm - 55.4) + 151
    else:
        return ((300 - 201) / (250.4 - 150.4)) * (pm - 150.4) + 201


# ---------------- LSTM MODEL ---------------- #
def lstm_forecast_with_evaluation(data, days=3):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.reshape(-1, 1))

    window = 14
    X, y = [], []

    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)

    y_pred = model.predict(X_test, verbose=0)

    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)

    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

    last_seq = scaled[-window:]
    future = []

    for _ in range(days):
        pred = model.predict(last_seq.reshape(1, window, 1), verbose=0)
        future.append(pred[0][0])
        last_seq = np.vstack((last_seq[1:], pred))

    future = scaler.inverse_transform(np.array(future).reshape(-1, 1))

    return future.flatten(), y_test_inv.flatten(), y_pred_inv.flatten(), mae, rmse


# ---------------- ROUTE ---------------- #
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        city = request.form["city"]

        lat, lon = get_lat_lon(city)
        if lat is None:
            return render_template("index.html", error="City not found")

        df = fetch_pm25(lat, lon)
        df["AQI"] = df["pm25"].apply(pm25_to_aqi)
        df["Category"] = df["AQI"].apply(aqi_category)

        future_aqi, y_actual, y_pred, mae, rmse = lstm_forecast_with_evaluation(
            df["AQI"].values, 3
        )

        today = date.today()
        future_dates = [today + timedelta(days=i) for i in range(1, 4)]

        # ---- Plot 1 ---- #
        plt.figure(figsize=(8,5))
        plt.plot(df["time"], df["AQI"], label="Historical AQI")
        plt.plot(future_dates, future_aqi, "o--", label="Forecast")
        plt.legend()
        plt.xticks(rotation=30)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("static/trend.png")
        plt.close()

        # ---- Plot 2 ---- #
        counts = df["Category"].value_counts()
        plt.figure(figsize=(8,5))
        plt.bar(counts.index, counts.values)
        plt.xticks(rotation=25)
        plt.ylabel("Days")
        plt.tight_layout()
        plt.savefig("static/distribution.png")
        plt.close()

        forecast_table = zip(
            [d.strftime("%Y-%m-%d") for d in future_dates],
            np.round(future_aqi, 1),
            [aqi_category(aqi) for aqi in future_aqi]
        )

        return render_template("index.html",
                               city=city,
                               forecast_table=forecast_table,
                               mae=round(mae,2),
                               rmse=round(rmse,2),
                               aqi_values_table=aqi_values_table)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
