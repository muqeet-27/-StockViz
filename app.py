import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.linear_model import LinearRegression

# =========================
# Load Environment Variables
# =========================
load_dotenv()

# =========================
# MongoDB Connection (User Data Only)
# =========================
def connect_to_mongo():
    uri = os.getenv("MONGO_URI")
    if not uri:
        st.sidebar.error("⚠️ MongoDB URI not found in environment variables.")
        return None
    
    client = MongoClient(uri, server_api=ServerApi('1'))
    
    try:
        client.admin.command('ping')
        st.sidebar.success("✅ Connected to MongoDB Atlas")
        return client
    except Exception as e:
        st.sidebar.error(f"❌ MongoDB connection failed: {e}")
        return None

# =========================
# Store user data in MongoDB
# =========================
def store_user_data(client, name, email, phone, age):
    if client:
        db = client["UserDB"]
        collection = db["users"]

        # Store user info
        user_info = {
            "name": name,
            "email": email,
            "phone": phone,
            "age": age,
            "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        collection.insert_one(user_info)
        st.sidebar.success("✅ User data saved to MongoDB")

# =========================
# Streamlit GUI Application
# =========================
st.title("📈 Stock Price Analysis & Comparison")
st.sidebar.header("🔧 Settings")

# =========================
# Sidebar Inputs
# =========================
tickers = st.sidebar.text_input("Enter stock symbols separated by commas (e.g., AAPL, MSFT, GOOGL)",value="nvda").upper().split(',')

# Dropdown for time range
time_range = st.sidebar.selectbox("Select Time Range", ["3 months", "6 months", "1 year", "3 years", "5 years"])

range_mapping = {
    "3 months": timedelta(days=90),
    "6 months": timedelta(days=180),
    "1 year": timedelta(days=365),
    "3 years": timedelta(days=3 * 365),
    "5 years": timedelta(days=5 * 365)
}

end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - range_mapping[time_range]).strftime('%Y-%m-%d')

# Connect to MongoDB
mongo_client = connect_to_mongo()

# =========================
# User Registration Form
# =========================
st.sidebar.header("👤 User Registration")

# User input fields
name = st.sidebar.text_input("Full Name")
email = st.sidebar.text_input("Email")
phone = st.sidebar.text_input("Phone Number")
age = st.sidebar.number_input("Age", min_value=1, max_value=100, step=1)

# Submit button for user registration
if st.sidebar.button("Register"):
    if name and email and phone and age:
        store_user_data(mongo_client, name, email, phone, age)
    else:
        st.sidebar.error("⚠️ Please fill all fields!")

# =========================
# Fetching Stock Data
# =========================
def get_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        return data, stock
    except Exception as e:
        st.sidebar.write(f"⚠️ Error fetching {ticker}: {e}")
        return pd.DataFrame(), None

# =========================
# Stock Price Prediction
# =========================
def predict_future_prices(data, days=30):
    # Prepare the data for Linear Regression
    data['Date'] = data.index
    data['Date'] = data['Date'].map(lambda x: x.toordinal())  # Convert dates to ordinal numbers
    X = data['Date'].values.reshape(-1, 1)  # Features (date as ordinal number)
    y = data['Close'].values  # Target (closing price)

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Predict future prices
    future_dates = [data['Date'].max() + i for i in range(1, days + 1)]
    future_dates_ordinal = np.array(future_dates).reshape(-1, 1)

    predicted_prices = model.predict(future_dates_ordinal)
    future_dates = [datetime.fromordinal(int(date)) for date in future_dates]

    return future_dates, predicted_prices

# =========================
# Plotting Functions
# =========================
def plot_line_chart(data, ticker):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Closing Price', color='blue')
    ax.set_title(f"{ticker} Historical Prices")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    st.pyplot(fig)

def plot_moving_averages(data, ticker):
    data['50_MA'] = data['Close'].rolling(window=50).mean()
    data['200_MA'] = data['Close'].rolling(window=200).mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Closing Price', color='blue')
    ax.plot(data.index, data['50_MA'], label='50-Day MA', color='orange')
    ax.plot(data.index, data['200_MA'], label='200-Day MA', color='green')
    ax.legend()
    st.pyplot(fig)

def plot_candlestick_chart(data, ticker):
    mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc)
    fig, ax = plt.subplots(figsize=(12, 6))
    mpf.plot(data, type='candle', style=s, ax=ax)
    st.pyplot(fig)

def plot_rsi(data, ticker):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, rsi, label='RSI', color='purple')
    ax.axhline(70, color='red', linestyle='--', label='Overbought')
    ax.axhline(30, color='green', linestyle='--', label='Oversold')
    ax.legend()
    st.pyplot(fig)

def plot_macd(data, ticker):
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, macd, label='MACD', color='blue')
    ax.plot(data.index, signal, label='Signal Line', color='orange')
    ax.legend()
    st.pyplot(fig)

# =========================
# Export Stock Data to CSV
# =========================
def export_to_csv(data, ticker):
    # Convert the dataframe to CSV
    csv = data.to_csv(index=True)  # Include index (date) in CSV file
    
    # Create a download button for CSV
    st.download_button(
        label=f"Download {ticker} Stock Data as CSV",
        data=csv,
        file_name=f"{ticker}_stock_data.csv",
        mime="text/csv"
    )

# =========================
# Main Display (No MongoDB Stock Saving)
# =========================
for ticker in tickers:
    ticker = ticker.strip().upper()
    data, stock = get_stock_data(ticker, start_date, end_date)

    if not data.empty:
        # Checkboxes to show different charts
        if st.sidebar.checkbox(f"Show {ticker} Historical Data"):
            plot_line_chart(data, ticker)
        if st.sidebar.checkbox(f"Show {ticker} Moving Averages"):
            plot_moving_averages(data, ticker)
        if st.sidebar.checkbox(f"Show {ticker} Candlestick Chart"):
            plot_candlestick_chart(data, ticker)
        if st.sidebar.checkbox(f"Show {ticker} RSI Indicator"):
            plot_rsi(data, ticker)
        if st.sidebar.checkbox(f"Show {ticker} MACD Indicator"):
            plot_macd(data, ticker)

        # Stock Price Prediction Checkbox
        if st.sidebar.checkbox(f"Predict {ticker} Future Prices"):
            future_dates, predicted_prices = predict_future_prices(data, days=30)

            # Plot historical data and predicted future prices
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index, data['Close'], label='Historical Prices', color='blue')
            ax.plot(future_dates, predicted_prices, label='Predicted Future Prices', color='orange', linestyle='--')
            ax.set_title(f"{ticker} Historical and Predicted Future Prices")
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (USD)')
            ax.legend()
            st.pyplot(fig)

        # Export Stock Data to CSV
        export_to_csv(data, ticker)

st.sidebar.write("👋 Made by Abdul Muqeet")
