import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pymongo import MongoClient, ASCENDING
from pymongo.server_api import ServerApi
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import bcrypt
import uuid
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
from logging.handlers import RotatingFileHandler
import time
from functools import wraps
from retrying import retry
import gzip
import smtplib
from email.message import EmailMessage
import urllib.parse
from apscheduler.schedulers.background import BackgroundScheduler
import matplotlib.pyplot as plt
import tempfile
from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =========================
# 🚀 Initial Setup
# =========================
load_dotenv()

# 📝 Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler = RotatingFileHandler('app.log', maxBytes=5*1024*1024, backupCount=3)
log_handler.setFormatter(log_formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

# ⏱️ Rate limiting decorator
def rate_limit(max_calls, time_frame):
    def decorator(func):
        calls = {}
        @wraps(func)
        def wrapper(*args, **kwargs):
            user_id = st.session_state.user["_id"] if st.session_state.authenticated else "guest"
            now = time.time()
            if user_id not in calls:
                calls[user_id] = []
            calls[user_id] = [call for call in calls[user_id] if call > now - time_frame]
            if len(calls[user_id]) >= max_calls:
                raise Exception(f"Rate limit exceeded. Max {max_calls} calls per {time_frame} seconds")
            calls[user_id].append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

# =========================
# 🗄️ MongoDB Connection
# =========================
@st.cache_resource
def connect_to_mongo(max_retries=3, delay=2):
    uri_template = os.getenv("MONGO_URI")
    if not uri_template:
        st.sidebar.error("⚠️ MongoDB URI not found.")
        return None
    try:
        if "://" in uri_template:
            protocol, rest = uri_template.split("://", 1)
            if "@" in rest:
                userinfo, hostinfo = rest.split("@", 1)
                if ":" in userinfo:
                    username, password = userinfo.split(":", 1)
                    encoded_username = urllib.parse.quote_plus(username)
                    encoded_password = urllib.parse.quote_plus(password)
                    uri = f"{protocol}://{encoded_username}:{encoded_password}@{hostinfo}"
                else:
                    uri = uri_template
            else:
                uri = uri_template
        else:
            uri = uri_template
        for attempt in range(max_retries):
            try:
                client = MongoClient(uri, server_api=ServerApi('1'))
                client.admin.command('ping')
                db = client["UserDB"]
                db.users.create_index([("email", ASCENDING)], unique=True)
                db.users.create_index([("verification_token", ASCENDING)], expireAfterSeconds=24*3600)
                db.users.create_index([("last_login_attempt", ASCENDING)])
                db.portfolios.create_index([("user_id", ASCENDING)], unique=True)
                db.watchlists.create_index([("user_id", ASCENDING)], unique=True)
                db.alerts.create_index([("user_id", ASCENDING), ("ticker", ASCENDING)])
                st.sidebar.success("✅ MongoDB Connected")
                return client
            except ConnectionFailure as e:
                logger.error(f"MongoDB connection attempt {attempt+1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                continue
        st.sidebar.error("❌ Failed to connect to MongoDB.")
        return None
    except Exception as e:
        logger.error(f"URI parsing error: {str(e)}")
        st.sidebar.error("❌ Invalid MongoDB URI format.")
        return None

# =========================
# 🛠️ Helper Functions
# =========================
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def is_valid_email(email):
    pattern = r"^(?=.{6,30}@gmail\.com$)[a-z0-9](?!.*\.\.)[a-z0-9._]*[a-z0-9]@gmail\.com$"
    return bool(re.match(pattern, email.lower()))

def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if len(password) > 128:
        return False, "Password too long"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r"\d", password):
        return False, "Password must contain at least one digit"
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character"
    return True, ""

def adjust_ticker(ticker, exchange):
    """Adjust ticker format based on the selected exchange."""
    if exchange == "US":
        return ticker
    elif exchange == "NSE":
        return f"{ticker}.NS"
    elif exchange == "BSE":
        return f"{ticker}.BO"
    return ticker

def reverse_adjust_ticker(ticker, exchange):
    """Remove exchange suffix to get the base ticker for storage or display."""
    if exchange == "NSE" and ticker.endswith(".NS"):
        return ticker[:-3]
    elif exchange == "BSE" and ticker.endswith(".BO"):
        return ticker[:-3]
    return ticker

@st.cache_data(ttl=3600)
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def get_company_name(ticker):
    stock = yf.Ticker(ticker)
    return stock.info.get("longName", ticker.split('.')[0] if '.' in ticker else ticker)

@st.cache_data(ttl=3600)
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def fetch_news(ticker, exchange):
    logger.info(f"Fetching news for {ticker} ({exchange}) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    try:
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            logger.error("News API key not found.")
            return []

        # Dynamically fetch the company name
        company_name = get_company_name(ticker)
        logger.info(f"Resolved company name for {ticker}: {company_name}")

        # Construct the query
        query = f"{company_name} India stock" if exchange in ["NSE", "BSE"] else company_name
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&language=en&sortBy=publishedAt&pageSize=5"
        
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        news_data = response.json()
        
        if news_data.get("status") != "ok":
            logger.error(f"News API error: {news_data.get('message', 'Unknown error')}")
            return []
        
        articles = news_data.get("articles", [])
        news_list = [
            {
                "title": article["title"],
                "url": article["url"],
                "source": article["source"]["name"],
                "publishedAt": pd.to_datetime(article["publishedAt"]).strftime('%Y-%m-%d %H:%M'),
                "summary": article.get("description", article["title"])[:250] + "..." if len(article.get("description", article["title"])) > 250 else article.get("description", article["title"])
            }
            for article in articles if article.get("title") and article.get("url")
        ]
        logger.info(f"Fetched {len(news_list)} news articles for {ticker}")
        return news_list
    
    except Exception as e:
        logger.error(f"Error fetching news for {ticker}: {str(e)}")
        return []

def ticker_for_alpha_vantage(ticker, exchange):
    """Format ticker for Alpha Vantage API."""
    if exchange == "US":
        return ticker
    elif exchange == "NSE":
        return f"{ticker}.NS"
    elif exchange == "BSE":
        return f"BSE:{ticker}"
    return ticker

def send_verification_email(email, token):
    app_url = os.getenv("APP_URL", "http://localhost:8501")
    verification_link = f"{app_url}/?token={token}"
    sender_email = os.getenv("SMTP_USER")
    sender_password = os.getenv("SMTP_PASS")
    smtp_server = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    try:
        if not all([sender_email, sender_password, smtp_server]):
            logger.error("SMTP configuration incomplete")
            return False, "SMTP configuration incomplete"
        msg = EmailMessage()
        msg['Subject'] = "Verify Your Email - TradeTrend Analyzer"
        msg['From'] = sender_email
        msg['To'] = email
        msg.set_content(f"""Welcome to TradeTrend Analyzer!
Please verify your email address by clicking this link:
{verification_link}
The link will expire in 24 hours.""")
        msg.add_alternative(f"""\
        <html>
            <body>
                <h2>Welcome to TradeTrend Analyzer!</h2>
                <p>Please verify your email address by clicking the button below:</p>
                <a href="{verification_link}" style="
                    background-color: #3b82f6;
                    color: white;
                    padding: 10px 20px;
                    text-decoration: none;
                    border-radius: 5px;
                    display: inline-block;
                    margin: 10px 0;
                ">Verify Email</a>
                <p>Or copy this link to your browser: {verification_link}</p>
                <p><small>The link will expire in 24 hours.</small></p>
            </body>
        </html>
        """, subtype='html')
        with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            logger.info(f"Email sent to {email} via SMTP")
            return True, "Email sent via SMTP"
    except Exception as e:
        logger.error(f"SMTP error: {str(e)}")
        return False, f"SMTP error: {str(e)}"

def send_alert_email(email, ticker, current_price, target_price):
    sender_email = os.getenv("SMTP_USER")
    sender_password = os.getenv("SMTP_PASS")
    smtp_server = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    try:
        msg = EmailMessage()
        msg['Subject'] = f"Price Alert for {ticker} - TradeTrend Analyzer"
        msg['From'] = sender_email
        msg['To'] = email
        msg.set_content(f"""Your price alert for {ticker} has been triggered!
Current Price: ${current_price:.2f}
Target Price: ${target_price:.2f}""")
        msg.add_alternative(f"""\
        <html>
            <body>
                <h2>Price Alert for {ticker}</h2>
                <p>Your alert has been triggered!</p>
                <p><strong>Current Price:</strong> ${current_price:.2f}</p>
                <p><strong>Target Price:</strong> ${target_price:.2f}</p>
            </body>
        </html>
        """, subtype='html')
        with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            logger.info(f"Alert email sent to {email} for {ticker}")
            return True
    except Exception as e:
        logger.error(f"Alert email error: {str(e)}")
        return False

# =========================
# 🔐 Authentication Functions
# =========================
def store_user_data(client, name, email, phone, age, password):
    if not client:
        return False
    db = client["UserDB"]
    if db.users.find_one({"email": email}):
        st.sidebar.error("⚠️ Email already registered")
        return False
    is_valid, msg = validate_password(password)
    if not is_valid:
        st.sidebar.error(f"⚠️ {msg}")
        return False
    hashed_pw = hash_password(password)
    verification_token = str(uuid.uuid4())
    user_data = {
        "name": name,
        "email": email,
        "phone": phone,
        "age": age,
        "password": hashed_pw,
        "verified": False,
        "verification_token": verification_token,
        "token_created": datetime.now(),
        "registered_at": datetime.now(),
        "login_attempts": 0,
        "last_login_attempt": None,
        "email_attempts": 0
    }
    try:
        result = db.users.insert_one(user_data)
        success, error_msg = send_verification_email(email, verification_token)
        if success:
            st.sidebar.success("✅ Registration successful! Check your email")
            return True
        else:
            st.sidebar.warning(f"⚠️ Registered but email failed: {error_msg}")
            return True
    except Exception as e:
        logger.error(f"User storage error: {str(e)}")
        st.sidebar.error("⚠️ Registration failed")
        return False

def verify_user(client, token):
    if not client:
        return False, "Database connection error"
    db = client["UserDB"]
    user = db.users.find_one({"verification_token": token})
    if not user:
        return False, "Invalid token"
    token_age = datetime.now() - user.get("token_created", datetime.now())
    if token_age > timedelta(hours=24):
        db.users.delete_one({"_id": user["_id"]})
        return False, "Token expired"
    try:
        db.users.update_one(
            {"_id": user["_id"]},
            {"$set": {"verified": True}, 
             "$unset": {"verification_token": "", "token_created": ""}}
        )
        return True, "Email verified!"
    except Exception as e:
        logger.error(f"Verification error: {str(e)}")
        return False, "Verification error"

@rate_limit(max_calls=5, time_frame=300)
def login_user(client, email, password):
    if not client:
        return None, "Database connection error"
    db = client["UserDB"]
    user = db.users.find_one({"email": email})
    if not user:
        return None, "Invalid credentials"
    if user.get("login_attempts", 0) >= 5:
        last_attempt = user.get("last_login_attempt")
        if last_attempt and (datetime.now() - last_attempt) < timedelta(minutes=15):
            return None, "Too many attempts. Try again later"
    if not check_password(password, user["password"]):
        db.users.update_one(
            {"_id": user["_id"]},
            {"$inc": {"login_attempts": 1},
             "$set": {"last_login_attempt": datetime.now()}}
        )
        return None, "Invalid credentials"
    if not user.get("verified", False):
        return None, "Email not verified"
    db.users.update_one(
        {"_id": user["_id"]},
        {"$set": {"login_attempts": 0, "last_login_attempt": None}}
    )
    return user, None

# =========================
# 📈 Stock Analysis Functions
# =========================
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def fetch_stock_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end)
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(ticker, start, end):
    try:
        return fetch_stock_data(ticker, start, end)
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        st.error(f"❌ Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_realtime_data(ticker, api_key, exchange):
    adjusted_ticker = ticker_for_alpha_vantage(ticker, exchange)
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={adjusted_ticker}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if "Global Quote" in data:
            return {
                "price": float(data["Global Quote"]["05. price"]),
                "change": float(data["Global Quote"]["09. change"]),
                "change_percent": data["Global Quote"]["10. change percent"],
                "timestamp": pd.to_datetime(data["Global Quote"]["07. latest trading day"])
            }
        return None
    except Exception as e:
        logger.error(f"Real-time data error for {adjusted_ticker}: {str(e)}")
        return None

def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def prepare_features(df):
    """Add technical indicators as features for the prediction model."""
    df = df.copy()
    # Moving Averages
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    # RSI
    df['RSI'] = calculate_rsi(df['Close'])
    # Volume
    df['Volume'] = df['Volume']
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df[['Close', 'MA50', 'MA200', 'RSI', 'Volume']]

def create_sequences(data, lookback):
    """Create sequences for LSTM training."""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, 0])  # Predict the 'Close' price
    return np.array(X), np.array(y)

def prepare_features(df):
    logger.info("Preparing features for DataFrame...")
    df = df.copy()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['Volume'] = df['Volume']
    df = df.dropna()
    logger.info(f"Features prepared, DataFrame size after dropna: {len(df)}")
    return df[['Close', 'MA50', 'RSI', 'Volume']]

def create_sequences(data, lookback):
    logger.info("Creating sequences for LSTM...")
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, 0])  # Predict the 'Close' price
    X, y = np.array(X), np.array(y)
    logger.info(f"Sequences created: X shape={X.shape}, y shape={y.shape}")
    return X, y

def predict_future_prices(df, days=30, lookback=30):
    logger.info(f"Starting prediction for {days} days with lookback {lookback}")
    logger.info(f"Input DataFrame size: {len(df)} rows")

    if df.empty or len(df) < lookback + 10:
        logger.warning(f"Insufficient data: {len(df)} rows. Need at least {lookback + 10} rows.")
        return [], []
    
    try:
        # Prepare features
        logger.info("Preparing features...")
        feature_df = prepare_features(df)
        data = feature_df.values
        logger.info(f"Feature data shape: {data.shape}")
        
        # Normalize the data
        logger.info("Normalizing data...")
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        logger.info(f"Scaled data shape: {scaled_data.shape}")
        
        # Create sequences for training
        logger.info("Creating sequences...")
        X, y = create_sequences(scaled_data, lookback)
        logger.info(f"Sequence shapes: X={X.shape}, y={y.shape}")
        
        # Split into train and test (90% train, 10% test)
        train_size = int(len(X) * 0.9)
        if train_size == 0:
            logger.warning("Not enough sequences for training after splitting.")
            raise ValueError("Not enough sequences for training after splitting.")
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        logger.info(f"Train/test split: X_train={X_train.shape}, X_test={X_test.shape}")
        
        # Build LSTM model
        logger.info("Building LSTM model...")
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(lookback, X.shape[2])),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16),
            Dense(1)
        ])
        
        logger.info("Compiling model...")
        model.compile(optimizer='adam', loss='mse')
        
        # Train the model
        logger.info("Training model...")
        model.fit(X_train, y_train, epochs=3, batch_size=16, verbose=0, validation_data=(X_test, y_test))
        logger.info("Model training completed.")
        
        # Prepare the last sequence for prediction
        last_sequence = scaled_data[-lookback:]
        future_preds = []
        
        current_sequence = last_sequence.copy()
        logger.info("Starting future predictions...")
        for i in range(days):
            current_sequence_reshaped = current_sequence.reshape((1, lookback, current_sequence.shape[1]))
            next_pred = model.predict(current_sequence_reshaped, verbose=0)
            future_preds.append(next_pred[0, 0])
            
            # Create a new row with the predicted close price and approximate other features
            next_row = np.zeros((1, current_sequence.shape[1]))
            next_row[0, 0] = next_pred[0, 0]
            next_row[0, 1:] = current_sequence[-1, 1:]
            current_sequence = np.vstack((current_sequence[1:], next_row))
        logger.info(f"Generated {len(future_preds)} future predictions.")
        
        # Inverse transform the predictions
        logger.info("Inverse transforming predictions...")
        future_preds_array = np.zeros((len(future_preds), data.shape[1]))
        future_preds_array[:, 0] = future_preds
        future_preds_transformed = scaler.inverse_transform(future_preds_array)
        future_prices = future_preds_transformed[:, 0]
        
        # Generate future dates
        logger.info("Generating future dates...")
        future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, days + 1)]
        
        logger.info("Prediction completed successfully.")
        return future_dates, future_prices.tolist()
    
    except Exception as e:
        logger.warning(f"LSTM prediction failed: {str(e)}. Falling back to Linear Regression.")
        df = df.copy()
        df['Date_Ordinal'] = df.index.map(datetime.toordinal)
        X = df['Date_Ordinal'].values.reshape(-1, 1)
        y = df['Close'].values
        model = LinearRegression().fit(X, y)
        last_date = df['Date_Ordinal'].max()
        future_dates_ordinal = np.array([last_date + i for i in range(1, days + 1)]).reshape(-1, 1)
        future_preds = model.predict(future_dates_ordinal)
        future_dates = [datetime.fromordinal(int(d)) for d in future_dates_ordinal.flatten()]
        logger.info("Fallback Linear Regression completed.")
        return future_dates, future_preds.tolist()
    
def plot_line(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title=f'{ticker} Closing Price', height=500, template='plotly_dark' if st.session_state.dark_mode else 'plotly')
    st.plotly_chart(fig, use_container_width=True)

def plot_ma(df, ticker):
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    df['200_MA'] = df['Close'].rolling(window=200).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
    fig.add_trace(go.Scatter(x=df.index, y=df['50_MA'], name='50 MA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['200_MA'], name='200 MA'))
    fig.update_layout(title=f'{ticker} Moving Averages', height=500, template='plotly_dark' if st.session_state.dark_mode else 'plotly')
    st.plotly_chart(fig, use_container_width=True)

def plot_candle(df, ticker):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close']
    )])
    fig.update_layout(title=f'{ticker} Candlestick Chart', height=600, template='plotly_dark' if st.session_state.dark_mode else 'plotly')
    st.plotly_chart(fig, use_container_width=True)

def plot_rsi(df):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple')))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(title='Relative Strength Index (RSI)', height=400, template='plotly_dark' if st.session_state.dark_mode else 'plotly')
    st.plotly_chart(fig, use_container_width=True)

def plot_macd(df):
    short_ema = df['Close'].ewm(span=12).mean()
    long_ema = df['Close'].ewm(span=26).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=macd, name='MACD', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=signal, name='Signal', line=dict(color='red')))
    fig.update_layout(title='MACD Indicator', height=400, template='plotly_dark' if st.session_state.dark_mode else 'plotly')
    st.plotly_chart(fig, use_container_width=True)

def plot_bollinger_bands(df):
    df['20_MA'] = df['Close'].rolling(window=20).mean()
    df['Upper'] = df['20_MA'] + 2 * df['Close'].rolling(window=20).std()
    df['Lower'] = df['20_MA'] - 2 * df['Close'].rolling(window=20).std()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
    fig.add_trace(go.Scatter(x=df.index, y=df['20_MA'], name='20 MA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], name='Upper Band', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], name='Lower Band', line=dict(color='green')))
    fig.update_layout(title='Bollinger Bands', height=500, template='plotly_dark' if st.session_state.dark_mode else 'plotly')
    st.plotly_chart(fig, use_container_width=True)

def plot_prediction(df, future_dates, future_preds, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Historical'))
    fig.add_trace(go.Scatter(x=future_dates, y=future_preds, name='Predicted', line=dict(dash='dash', color='orange')))
    fig.update_layout(title=f"{ticker} Price Forecast", height=500, template='plotly_dark' if st.session_state.dark_mode else 'plotly')
    st.plotly_chart(fig, use_container_width=True)

# =========================
# 💼 Portfolio & Watchlist Functions
# =========================
def get_portfolio(client, user_id):
    if not client:
        return []
    db = client["UserDB"]
    portfolio = db.portfolios.find_one({"user_id": user_id})
    return portfolio.get("holdings", []) if portfolio else []

def update_portfolio(client, user_id, holdings):
    if not client:
        return False
    with st.spinner("Updating portfolio..."):
        if not isinstance(holdings, list):
            st.error("Invalid holdings format")
            return False
        db = client["UserDB"]
        try:
            db.portfolios.update_one(
                {"user_id": user_id},
                {"$set": {"holdings": holdings, "updated_at": datetime.now()}},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Portfolio update error: {str(e)}")
            return False

def get_watchlist(client, user_id):
    if not client:
        return []
    db = client["UserDB"]
    watchlist = db.watchlists.find_one({"user_id": user_id})
    return watchlist.get("tickers", []) if watchlist else []

def update_watchlist(client, user_id, tickers):
    if not client:
        return False
    with st.spinner("Updating watchlist..."):
        if not isinstance(tickers, list):
            st.error("Invalid watchlist format")
            return False
        db = client["UserDB"]
        try:
            db.watchlists.update_one(
                {"user_id": user_id},
                {"$set": {"tickers": tickers, "updated_at": datetime.now()}},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Watchlist update error: {str(e)}")
            return False

@st.cache_data(ttl=300)
def calculate_portfolio_value(holdings, exchange):
    total_value = 0
    value_data = []
    errors = []
    for item in holdings:
        try:
            base_ticker = item["ticker"]
            ticker = adjust_ticker(base_ticker, exchange)
            shares = item["shares"]
            stock = yf.Ticker(ticker)
            history = stock.history(period="1d")
            if history.empty:
                history = stock.history(period="5d")
                if history.empty:
                    raise ValueError("No recent data available; market might be closed or ticker is invalid.")
            current_price = history["Close"].iloc[-1]
            value = shares * current_price
            total_value += value
            value_data.append({"Ticker": base_ticker, "Shares": shares, "Value": value, "Price": current_price})
        except Exception as e:
            error_msg = f"Failed to fetch data for {ticker}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
    return total_value, value_data, errors

# =========================
# 🔔 Price Alerts
# =========================
def add_alert(client, user_id, ticker, target_price, condition, exchange):
    if not client:
        return False
    db = client["UserDB"]
    alert = {
        "user_id": user_id,
        "ticker": ticker,
        "exchange": exchange,
        "target_price": target_price,
        "condition": condition,
        "status": "active",
        "created_at": datetime.now()
    }
    try:
        db.alerts.insert_one(alert)
        return True
    except Exception as e:
        logger.error(f"Alert creation error: {str(e)}")
        return False

def check_alerts():
    client = connect_to_mongo()
    if not client:
        return
    db = client["UserDB"]
    alerts = db.alerts.find({"status": "active"})
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    for alert in alerts:
        base_ticker = alert["ticker"]
        exchange = alert.get("exchange", "US")
        adjusted_ticker = adjust_ticker(base_ticker, exchange)
        data = fetch_realtime_data(base_ticker, api_key, exchange)
        if data:
            price = data["price"]
            if (alert["condition"] == "above" and price >= alert["target_price"]) or \
               (alert["condition"] == "below" and price <= alert["target_price"]):
                user = db.users.find_one({"_id": alert["user_id"]})
                send_alert_email(user["email"], adjusted_ticker, price, alert["target_price"])
                db.alerts.update_one({"_id": alert["_id"]}, {"$set": {"status": "triggered"}})

scheduler = BackgroundScheduler()
scheduler.add_job(check_alerts, 'interval', minutes=5)
scheduler.start()

# =========================
# 📅 Economic Calendar
# =========================
@st.cache_data(ttl=86400)
def fetch_economic_calendar(api_key):
    url = f"https://www.alphavantage.co/query?function=ECONOMIC_CALENDAR&apikey={api_key}"
    try:
        response = requests.get(url)
        events = response.json().get("data", [])
        return pd.DataFrame(events)[:10]
    except Exception as e:
        logger.error(f"Economic calendar error: {str(e)}")
        return pd.DataFrame()

# =========================
# 📉 Risk Analysis
# =========================
def calculate_risk_metrics(holdings, exchange, benchmark="^GSPC"):
    if not holdings:
        return None
    if exchange == "NSE":
        benchmark = "^NSEI"
    elif exchange == "BSE":
        benchmark = "^BSESN"
    else:
        benchmark = "^GSPC"
    portfolio_values = []
    benchmark_data = get_stock_data(benchmark, datetime.now() - timedelta(days=365), datetime.now())
    returns = []
    for item in holdings:
        base_ticker = item["ticker"]
        ticker = adjust_ticker(base_ticker, exchange)
        shares = item["shares"]
        df = get_stock_data(ticker, datetime.now() - timedelta(days=365), datetime.now())
        if not df.empty:
            prices = df["Close"] * shares
            portfolio_values.append(prices)
            returns.append(df["Close"].pct_change().dropna())
    if not portfolio_values:
        return None
    portfolio_prices = pd.concat(portfolio_values, axis=1).sum(axis=1)
    portfolio_returns = portfolio_prices.pct_change().dropna()
    benchmark_returns = benchmark_data["Close"].pct_change().dropna()
    mean = portfolio_returns.mean()
    std = portfolio_returns.std()
    var_95 = norm.ppf(0.05, mean, std) * portfolio_prices.iloc[-1]
    aligned_returns = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    cov = np.cov(aligned_returns.iloc[:, 0], aligned_returns.iloc[:, 1])[0][1]
    var_bm = np.var(benchmark_returns)
    beta = cov / var_bm if var_bm != 0 else 0
    rf_rate = 0.02
    sharpe = (portfolio_returns.mean() - rf_rate/252) / portfolio_returns.std() * np.sqrt(252)
    return {"VaR_95": -var_95, "Beta": beta, "Sharpe_Ratio": sharpe}

# =========================
# 🤖 Stock Recommendations
# =========================
def get_stock_recommendations(sector="Technology", risk_tolerance="Moderate", exchange="US"):
    tickers_dict = {
        "US": {
            "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "Finance": ["JPM", "BAC", "WFC", "GS", "MS"],
            "Healthcare": ["PFE", "JNJ", "MRK", "ABBV", "GILD"]
        },
        "NSE": {
            "Technology": ["INFY", "TCS", "WIPRO", "HCLTECH", "TECHM"],
            "Finance": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK"],
            "Healthcare": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN"]
        },
        "BSE": {
            "Technology": ["INFY", "TCS", "WIPRO", "HCLTECH", "TECHM"],
            "Finance": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK"],
            "Healthcare": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN"]
        }
    }
    tickers = tickers_dict.get(exchange, {}).get(sector, tickers_dict["US"]["Technology"])
    adjusted_tickers = [adjust_ticker(ticker, exchange) for ticker in tickers]
    recommendations = []
    for ticker in adjusted_tickers:
        df = get_stock_data(ticker, datetime.now() - timedelta(days=90), datetime.now())
        if not df.empty:
            momentum = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1
            if momentum > 0.05:
                base_ticker = reverse_adjust_ticker(ticker, exchange)
                recommendations.append({"ticker": base_ticker, "momentum": momentum})
    return sorted(recommendations, key=lambda x: x["momentum"], reverse=True)[:3]

# =========================
# 📱 PDF Export
# =========================
def generate_pdf_report(holdings, watchlist, portfolio_value, risk_metrics, exchange):
    html_content = f"""
    <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #1e40af; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>TradeTrend Analyzer Report</h1>
            <h2>Portfolio Summary</h2>
            <p>Total Value: ${portfolio_value:.2f}</p>
            <table>
                <tr><th>Ticker</th><th>Shares</th><th>Value</th></tr>
                {''.join([f'<tr><td>{h["ticker"]}</td><td>{h["shares"]}</td><td>${h["shares"] * yf.Ticker(adjust_ticker(h["ticker"], exchange)).history(period="1d")["Close"].iloc[-1]:.2f}</td></tr>' for h in holdings])}
            </table>
            <h2>Risk Metrics</h2>
            <p>VaR (95%): ${risk_metrics.get('VaR_95', 0):.2f}</p>
            <p>Beta: {risk_metrics.get('Beta', 0):.2f}</p>
            <h2>Watchlist</h2>
            <ul>{''.join([f'<li>{t}</li>' for t in watchlist])}</ul>
        </body>
    </html>
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        tmp.write(html_content.encode())
        tmp_path = tmp.name
    pdf_path = tmp_path.replace(".html", ".pdf")
    HTML(tmp_path).write_pdf(pdf_path)
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
    os.remove(tmp_path)
    os.remove(pdf_path)
    return pdf_data

# =========================
# 🌍 Multi-Currency Support
# =========================
@st.cache_data(ttl=3600)
def get_exchange_rate(base_currency, target_currency):
    url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
    try:
        response = requests.get(url)
        return response.json()["rates"].get(target_currency, 1.0)
    except Exception as e:
        logger.error(f"Exchange rate error: {str(e)}")
        return 1.0

# =========================
# 🎨 Frontend Styling
# =========================
def apply_css_styles(dark_mode=True):
    css = """
    <style>
        @import url('https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css');
        .stApp {
            background-color: %s;
            font-family: 'Inter', sans-serif;
        }
        .sidebar .stSidebar {
            background-color: %s;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .stButton button {
            background-color: #3b82f6;
            color: white;
            border-radius: 0.375rem;
            padding: 0.5rem 1rem;
            transition: background-color 0.3s;
        }
        .stButton button:hover {
            background-color: #2563eb;
        }
        .stTextInput input, .stNumberInput input, .stSelectbox select, .stTextArea textarea {
            background-color: %s;
            color: %s;
            border: 1px solid %s;
            border-radius: 0.375rem;
            padding: 0.5rem;
        }
        .news-item {
            border-bottom: 1px solid %s;
            padding: 1rem 0;
            transition: background-color 0.2s;
        }
        .news-item:hover {
            background-color: %s;
        }
        .stTabs [role="tab"] {
            background-color: %s;
            color: %s;
            border-radius: 0.375rem 0.375rem 0 0;
            padding: 0.5rem 1rem;
        }
        .stTabs [role="tab"][aria-selected="true"] {
            background-color: #3b82f6;
            color: white;
        }
        @media (max-width: 600px) {
            .stApp { padding: 0.5rem; }
            .stSidebar { width: 100%% !important; }
            .stButton button { width: 100%%; }
            .news-item { font-size: 0.875rem; }
        }
        [data-testid="stHeader"] {
            background-color: %s;
        }
    </style>
    """
    if dark_mode:
        st.markdown(css % (
            '#111827', '#1f2937', '#374151', '#f3f4f6', '#4b5563',
            '#4b5563', '#1f2937', '#374151', '#f3f4f6', '#1f2937'
        ), unsafe_allow_html=True)
    else:
        st.markdown(css % (
            '#ffffff', '#f3f4f6', '#f3f4f6', '#1f2937', '#d1d5db',
            '#d1d5db', '#f9fafb', '#e5e7eb', '#1f2937', '#f3f4f6'
        ), unsafe_allow_html=True)

# =========================
# 🧠 Session State
# =========================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.show_logout_message = False
    st.session_state.show_verification_message = False
    st.session_state.dark_mode = True
    st.session_state.currency = "USD"
    st.session_state.exchange = "US"

# =========================
# ✅ Verification Handler
# =========================
query_params = st.query_params
if 'token' in query_params:
    token = query_params['token']
    client = connect_to_mongo()
    success, message = verify_user(client, token)
    if success:
        st.session_state.show_verification_message = True
        st.session_state.verification_message = message
    else:
        st.error(message)
    st.query_params.clear()

# =========================
# 📏 Sidebar
# =========================
st.sidebar.header("🔐 Authentication")
mongo_client = connect_to_mongo()

with st.sidebar.container():
    st.session_state.dark_mode = st.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode)
    apply_css_styles(st.session_state.dark_mode)
    currency = st.selectbox("💱 Currency", ["USD", "EUR", "GBP", "JPY", "INR"], key="currency")
    exchange = st.selectbox("🏛️ Stock Exchange", ["US", "NSE", "BSE"], key="exchange")
    exchange_rate = get_exchange_rate("USD", currency)

auth_mode = st.sidebar.radio("Choose Action", ["Login", "Register"], key="auth_mode")

if auth_mode == "Register":
    with st.sidebar.form("register_form"):
        name = st.text_input("Name", key="reg_name")
        email = st.text_input("Email", key="reg_email").lower()
        phone = st.text_input("Phone", key="reg_phone")
        age = st.number_input("Age", min_value=1, max_value=100, key="reg_age")
        password = st.text_input("Password", type="password", key="reg_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
        if st.form_submit_button("Register"):
            if not all([name, email, phone, age, password, confirm_password]):
                st.sidebar.error("⚠️ All fields required")
            elif not is_valid_email(email):
                st.sidebar.error("⚠️ Invalid email format")
            elif password != confirm_password:
                st.sidebar.error("⚠️ Passwords don't match")
            else:
                with st.spinner("Registering..."):
                    if store_user_data(mongo_client, name, email, phone, age, password):
                        st.sidebar.info("📧 Check your email for verification")

elif auth_mode == "Login":
    with st.sidebar.form("login_form"):
        login_email = st.text_input("Email", key="login_email").lower()
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.form_submit_button("Login"):
            try:
                user, error = login_user(mongo_client, login_email, login_password)
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user = user
                    st.session_state.show_logout_message = False
                    st.sidebar.success(f"✅ Welcome, {user['name']}!")
                else:
                    st.sidebar.error(f"❌ {error}")
                    if "not verified" in error.lower():
                        if st.button("Resend Verification Email"):
                            user = mongo_client["UserDB"]["users"].find_one({"email": login_email})
                            if user and not user.get("verified", False):
                                last_sent = user.get("token_created")
                                if last_sent and (datetime.now() - last_sent) < timedelta(minutes=5):
                                    st.sidebar.warning("Email sent recently. Please wait")
                                else:
                                    token = str(uuid.uuid4())
                                    mongo_client["UserDB"]["users"].update_one(
                                        {"_id": user["_id"]},
                                        {"$set": {"verification_token": token, "token_created": datetime.now()},
                                         "$inc": {"email_attempts": 1}}
                                    )
                                    success, error_msg = send_verification_email(login_email, token)
                                    if success:
                                        st.sidebar.success("✅ Email resent")
                                    else:
                                        st.sidebar.error(f"Failed to resend: {error_msg}")
            except Exception as e:
                st.sidebar.error(f"Login error: {str(e)}")

if st.session_state.authenticated:
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.show_logout_message = True
        st.rerun()

# =========================
# 🖥️ Main Content
# =========================
if st.session_state.show_verification_message:
    st.success(st.session_state.verification_message)
    st.session_state.show_verification_message = False

if st.session_state.authenticated:
    st.title("📊 TradeTrend Analyzer")
    tabs = st.tabs(["Stock Analysis", "Portfolio", "Recommendations", "Economic Calendar", "Education"])
    
    with tabs[0]:
        st.header("📈 Stock Analysis")
        with st.container():
            col1, col2 = st.columns([3, 1])
            base_tickers = col1.text_input("Enter symbols (e.g., AAPL,MSFT or RELIANCE,TCS)", value="").upper().split(',')
            tickers = [adjust_ticker(ticker.strip(), exchange) for ticker in base_tickers if ticker.strip()]
            data_source = col2.selectbox("Data Source", ["Historical (yFinance)", "Real-Time (Alpha Vantage)"], key="data_source")
        
        time_range_option = st.selectbox(
            "Time Range", ["3 months", "6 months", "1 year", "3 years", "5 years", "Custom Range"],
            key="time_range"
        )
        start_date = end_date = None
        if time_range_option != "Custom Range":
            range_mapping = {
                "3 months": timedelta(days=90),
                "6 months": timedelta(days=180),
                "1 year": timedelta(days=365),
                "3 years": timedelta(days=3*365),
                "5 years": timedelta(days=5*365)
            }
            end_date = datetime.today()
            start_date = end_date - range_mapping[time_range_option]
        else:
            col1, col2 = st.columns(2)
            from_date = col1.date_input("From Date")
            to_date = col2.date_input("To Date")
            today = datetime.today().date()
            if from_date >= today or to_date >= today:
                st.warning("⚠️ Select past dates")
            elif from_date > to_date:
                st.warning("⚠️ Invalid date range")
            else:
                start_date = from_date
                end_date = to_date
        
        if start_date and end_date:
            for ticker in tickers:
                base_ticker = reverse_adjust_ticker(ticker, exchange)
                if not ticker:
                    continue
                if data_source == "Real-Time (Alpha Vantage)":
                    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
                    if not api_key:
                        st.error("⚠️ Alpha Vantage API key not configured")
                    else:
                        data = fetch_realtime_data(base_ticker, api_key, exchange)
                        if data:
                            st.markdown(f"""
                            <div class="p-4 bg-gray-800 rounded-lg">
                                <h3 class="text-lg font-semibold">{base_ticker}</h3>
                                <p>Price: {currency} {(data['price'] * exchange_rate):.2f}</p>
                                <p>Change: {data['change_percent']}</p>
                                <p>Last Updated: {data['timestamp']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            if st.button(f"Refresh {base_ticker}", key=f"refresh_{base_ticker}"):
                                st.cache_data.clear()
                                st.rerun()
                else:
                    with st.spinner(f"Loading {base_ticker} data..."):
                        df = get_stock_data(ticker, start_date, end_date)
                    if not df.empty:
                        st.markdown(f"### 📍 {base_ticker} Analysis")
                        show_line = st.checkbox(f"📈 Line Chart", key=f"line_{base_ticker}", value=True)
                        show_ma = st.checkbox(f"📊 Moving Averages", key=f"ma_{base_ticker}")
                        show_candle = st.checkbox(f"💹 Candlestick", key=f"candle_{base_ticker}")
                        show_rsi = st.checkbox(f"📈 RSI", key=f"rsi_{base_ticker}")
                        show_macd = st.checkbox(f"📉 MACD", key=f"macd_{base_ticker}")
                        show_bollinger = st.checkbox(f"📊 Bollinger Bands", key=f"bollinger_{base_ticker}")
                        show_prediction = st.checkbox(f"🧠 Prediction", key=f"pred_{base_ticker}")
                        if show_line:
                            plot_line(df, base_ticker)
                        if show_ma:
                            plot_ma(df, base_ticker)
                        if show_candle:
                            plot_candle(df, base_ticker)
                        if show_rsi:
                            plot_rsi(df)
                        if show_macd:
                            plot_macd(df)
                        if show_bollinger:
                            plot_bollinger_bands(df)
                        if show_prediction:
                            dates, preds = predict_future_prices(df)
                            if dates and preds:
                                plot_prediction(df, dates, preds, base_ticker)
                            else:
                                st.warning(f"⚠️ Unable to generate predictions for {base_ticker}")
                        with st.expander(f"📰 News for {base_ticker}"):
                            news_items = fetch_news(ticker, exchange)
                            logger.info(f"Displaying {len(news_items)} news items for {ticker}")
                            if news_items:
                                for item in news_items:
                                    st.markdown(f"""
                                    <div class="news-item">
                                        <h3 class="text-md font-medium">{item['title']}</h3>
                                        <p class="text-sm text-gray-400">{item['source']} • {item['publishedAt']}</p>
                                        <p class="text-sm">{item['summary']}</p>
                                        <a href="{item['url']}" target="_blank" class="text-blue-500 hover:underline">Read more</a>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info(f"No recent news for {base_ticker}. Check [Live Mint](https://www.livemint.com/market/stock-market-news) for updates.")
                        compressed_csv = gzip.compress(df.to_csv(index=True).encode('utf-8'))
                        st.download_button(f"📥 Download {base_ticker} Data", compressed_csv, f"{base_ticker}_data.csv.gz", "application/gzip")

    with tabs[1]:
        st.header("💼 Portfolio")
        holdings = get_portfolio(mongo_client, st.session_state.user["_id"])
        with st.container():
            col1, col2 = st.columns([2, 1])
            base_ticker = col1.text_input("Ticker", key="portfolio_ticker").upper()
            ticker = adjust_ticker(base_ticker, exchange)
            shares = col2.number_input("Shares", min_value=0.0, value=1.0, step=0.1)
            if st.button("Add to Portfolio"):
                if base_ticker:
                    holdings.append({"ticker": base_ticker, "shares": shares})
                    if update_portfolio(mongo_client, st.session_state.user["_id"], holdings):
                        st.success("Portfolio updated!")
        if holdings:
            total_value, value_data, errors = calculate_portfolio_value(holdings, exchange)
            if errors:
                for error in errors:
                    st.warning(error)
            if value_data:
                st.markdown(f"**Total Value**: {currency} {(total_value * exchange_rate):.2f}")
                df_portfolio = pd.DataFrame(value_data)
                st.dataframe(df_portfolio)
                if not df_portfolio.empty:
                    fig = go.Figure(data=[go.Pie(labels=df_portfolio["Ticker"], values=df_portfolio["Value"])])
                    fig.update_layout(title="Portfolio Allocation", template='plotly_dark' if st.session_state.dark_mode else 'plotly')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No portfolio data available to display allocation chart.")
            else:
                st.info("Unable to fetch portfolio data. Please check the tickers or try again later.")
            for item in holdings:
                st.markdown(f"{item['ticker']}: {item['shares']} shares")
                if st.button(f"Remove {item['ticker']}", key=f"remove_{item['ticker']}"):
                    holdings = [h for h in holdings if h['ticker'] != item['ticker']]
                    update_portfolio(mongo_client, st.session_state.user["_id"], holdings)
                    st.rerun()
            risk_metrics = calculate_risk_metrics(holdings, exchange)
            if risk_metrics:
                st.markdown("### 📉 Risk Metrics")
                st.write(f"**VaR (95% Confidence, 1-Day)**: {currency} {(risk_metrics['VaR_95'] * exchange_rate):.2f}")
                st.write(f"**Portfolio Beta**: {risk_metrics['Beta']:.2f}")
                st.write(f"**Sharpe Ratio**: {risk_metrics['Sharpe_Ratio']:.2f}")
            if value_data and st.button("Export Portfolio Report"):
                watchlist = get_watchlist(mongo_client, st.session_state.user["_id"])
                pdf_data = generate_pdf_report(holdings, watchlist, total_value, risk_metrics, exchange)
                st.download_button("Download PDF Report", pdf_data, "portfolio_report.pdf", "application/pdf")
        with st.expander("🔔 Alerts"):
            alerts = mongo_client["UserDB"]["alerts"].find({"user_id": st.session_state.user["_id"], "status": "active"})
            st.write("Active Alerts:")
            for alert in alerts:
                alert_exchange = alert.get("exchange", "US")
                adjusted_ticker = adjust_ticker(alert["ticker"], alert_exchange)
                st.write(f"{alert['ticker']} ({alert_exchange}): {alert['condition'].capitalize()} {currency} {(alert['target_price'] * exchange_rate):.2f}")
                if st.button(f"Delete {alert['ticker']}", key=f"delete_alert_{alert['_id']}"):
                    mongo_client["UserDB"]["alerts"].delete_one({"_id": alert["_id"]})
                    st.rerun()
            alert_base_ticker = st.text_input("Alert Ticker", key="alert_ticker").upper()
            target_price = st.number_input("Target Price", min_value=0.0, step=0.01, key="alert_price")
            condition = st.selectbox("Condition", ["Above", "Below"], key="alert_condition")
            if st.button("Set Alert"):
                if add_alert(mongo_client, st.session_state.user["_id"], alert_base_ticker, target_price, condition.lower(), exchange):
                    st.success("Alert set!")
        with st.expander("👀 Watchlist"):
            watchlist = get_watchlist(mongo_client, st.session_state.user["_id"])
            watch_base_ticker = st.text_input("Add Ticker", key="watchlist_ticker").upper()
            if st.button("Add to Watchlist"):
                if watch_base_ticker and watch_base_ticker not in watchlist:
                    watchlist.append(watch_base_ticker)
                    if update_watchlist(mongo_client, st.session_state.user["_id"], watchlist):
                        st.success("Watchlist updated!")
            if watchlist:
                st.write("Watched Tickers:")
                for base_ticker in watchlist:
                    col1, col2 = st.columns([3, 1])
                    col1.write(base_ticker)
                    if col2.button("✕", key=f"remove_watch_{base_ticker}"):
                        watchlist.remove(base_ticker)
                        update_watchlist(mongo_client, st.session_state.user["_id"], watchlist)
                        st.rerun()

    with tabs[2]:
        st.header("🤖 Stock Recommendations")
        sector = st.selectbox("Sector", ["Technology", "Finance", "Healthcare"], key="rec_sector")
        risk_tolerance = st.selectbox("Risk Tolerance", ["Low", "Moderate", "High"], key="rec_risk")
        if st.button("Get Recommendations"):
            recs = get_stock_recommendations(sector, risk_tolerance, exchange)
            if recs:
                for rec in recs:
                    st.markdown(f"**{rec['ticker']}**: Momentum Score {rec['momentum']:.2%}")
            else:
                st.info("No recommendations available")

    with tabs[3]:
        st.header("📅 Economic Calendar")
        calendar = fetch_economic_calendar(os.getenv("ALPHA_VANTAGE_API_KEY"))
        if not calendar.empty:
            st.dataframe(calendar[["date", "event", "impact", "country"]])
        else:
            st.info("No upcoming events found")

    with tabs[4]:
        st.header("📚 Learn About Investing")
        st.markdown("""
        ### Technical Analysis
        - [Understanding RSI](https://www.investopedia.com/terms/r/rsi.asp)
        - [MACD Explained](https://www.youtube.com/watch?v=kA9hK4G4fTE)
        ### Portfolio Management
        - [Diversification Strategies](https://www.investopedia.com/terms/d/diversification.asp)
        ### Videos
        <iframe width="100%" height="315" src="https://www.youtube.com/embed/kA9hK4G4fTE" frameborder="0" allowfullscreen></iframe>
        """, unsafe_allow_html=True)

elif st.session_state.show_logout_message:
    st.markdown("""
    <div class="text-center mt-12 text-2xl">
        👋 Thank you for using TradeTrend Analyzer!
    </div>
    """, unsafe_allow_html=True)
    st.session_state.show_logout_message = False
else:
    st.markdown("""
    <div class="text-center mt-12 text-2xl">
        🔒 Please log in to access TradeTrend Analyzer
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Made by **Abdul Muqeet**")
st.sidebar.markdown("CSE Dept , GMU")
