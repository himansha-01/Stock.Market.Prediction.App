import gradio as gr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
import joblib
import requests
import time
import datetime
import logging
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "lstm_stock_model.h5"
SCALER_PATH = "scaler.pkl"
PLOT_PATH = "stock_plot.png"
FMP_API_KEY = "5x9eFLfTuEXnryKKD27sKz01W0pRClUd"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# Function to save the LSTM model
def save_lstm_model(model, scaler, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    try:
        logger.info(f"Saving model to {model_path} and scaler to {scaler_path}")
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        return True
    except Exception as e:
        logger.error(f"Error saving model/scaler: {e}")
        return False

# Function to load model and scaler
def load_lstm_model_and_scaler():
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            logger.info(f"Loading model from {MODEL_PATH} and scaler from {SCALER_PATH}")
            model = load_model(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            return model, scaler, "Model loaded successfully"
        else:
            logger.warning("Model or scaler files not found")
            return None, None, "Model files not found"
    except Exception as e:
        logger.error(f"Error loading model/scaler: {e}")
        return None, None, f"Error loading model: {e}"

# FMP API functions
@lru_cache(maxsize=128)
def get_fmp_historical_prices(ticker, limit=730):
    """
    Fetch historical daily prices from FMP API
    Limit parameter gets n days of data (730 days = ~2 years)
    Using lru_cache to avoid redundant API calls
    """
    url = f"{FMP_BASE_URL}/historical-price-full/{ticker}?apikey={FMP_API_KEY}&serietype=line&limit={limit}"
    logger.info(f"Fetching historical data for {ticker} (limit={limit})")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors
        data = response.json()
        
        if 'historical' not in data:
            logger.warning(f"No historical data found for {ticker}")
            return None, f"No historical data available for {ticker}"
        
        # Convert to DataFrame and sort by date
        historical_data = pd.DataFrame(data['historical'])
        historical_data['date'] = pd.to_datetime(historical_data['date'])
        historical_data = historical_data.sort_values('date')
        historical_data.set_index('date', inplace=True)
        
        logger.info(f"Successfully fetched {len(historical_data)} records for {ticker}")
        return historical_data, None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from FMP: {e}")
        return None, f"API request error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error processing FMP data: {e}")
        return None, f"Error processing data: {e}"

# Get USD to INR conversion rate
@lru_cache(maxsize=1)  # Cache for 1 hour
def get_usd_inr_rate():
    try:
        # Using FMP for exchange rate as well
        url = f"{FMP_BASE_URL}/fx/USDINR?apikey={FMP_API_KEY}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                return data[0].get('price', 75.0)
        
        # Fallback to alternative API
        fallback_url = "https://api.exchangerate-api.com/v4/latest/USD"
        fallback_response = requests.get(fallback_url, timeout=5)
        if fallback_response.status_code == 200:
            fallback_data = fallback_response.json()
            return fallback_data["rates"].get("INR", 75.0)
            
        return 75.0  # Hardcoded fallback
    except Exception as e:
        logger.error(f"Error fetching exchange rate: {e}")
        return 75.0

# Function to get historical data for training
def get_historical_data(ticker, days=730):
    data, error = get_fmp_historical_prices(ticker, limit=days)
    if data is not None and not data.empty:
        return data["close"].values, None
    return None, error or "Failed to get historical data"

# Function to train LSTM model
def train_lstm_model(data, ticker="AAPL"):
    try:
        logger.info(f"Training LSTM model for {ticker} with {len(data)} data points")
        
        # Handle NaN values
        data = data[~np.isnan(data)]
        if len(data) < 100:
            return None, None, "Insufficient data for training"
            
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))

        # Create sequences for LSTM (adjust sequence length based on data size)
        seq_length = min(60, len(scaled_data) // 3)
        X_train, y_train = [], []
        for i in range(seq_length, len(scaled_data)):
            X_train.append(scaled_data[i-seq_length:i, 0])
            y_train.append(scaled_data[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

        success = save_lstm_model(model, scaler)
        if success:
            return model, scaler, f"Model trained successfully for {ticker}"
        else:
            return None, None, "Failed to save model"
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return None, None, f"Error during model training: {e}"

# Initialize or load model
def initialize_model(ticker="AAPL"):
    # Try to load existing model
    model, scaler, status = load_lstm_model_and_scaler()
    
    # If loading fails, train a new model
    if model is None or scaler is None:
        logger.info("No model found or loading failed. Training new model...")
        data, error = get_historical_data(ticker)
        if data is not None:
            model, scaler, status = train_lstm_model(data, ticker)
            return model, scaler, status
        else:
            return None, None, f"Failed to initialize model: {error}"
    
    return model, scaler, status

# Plot stock data with improved visuals
def plot_stock_data(ticker):
    # Fetch data
    data, error = get_fmp_historical_prices(ticker, limit=30)  # Last 30 days
    if error:
        return None, error
    
    if data is None or data.empty:
        return None, "No data available for this ticker."
    
    try:
        # Create a more visually appealing plot
        plt.figure(figsize=(12, 6))
        plt.style.use('ggplot')
        
        # Plot close price
        plt.plot(data.index, data["close"], label=f"{ticker} Close Price", color='#1f77b4', linewidth=2.5)
        
        # Add moving averages
        if len(data) > 5:
            ma5 = data["close"].rolling(window=5).mean()
            plt.plot(data.index, ma5, label="5-Day MA", color='#ff7f0e', linestyle='--', linewidth=1.5)
        
        if len(data) > 20:
            ma20 = data["close"].rolling(window=20).mean()
            plt.plot(data.index, ma20, label="20-Day MA", color='#2ca02c', linestyle='--', linewidth=1.5)
        
        # Format the plot
        plt.title(f"{ticker} Stock Price", fontsize=16, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price (USD)", fontsize=12)
        plt.legend(loc="best", frameon=True)
        plt.grid(True, alpha=0.3)
        
        # Add current price annotation
        current_price = data["close"].iloc[-1]
        plt.annotate(f'${current_price:.2f}', 
                    xy=(data.index[-1], current_price),
                    xytext=(data.index[-1], current_price*1.05),
                    fontsize=12,
                    arrowprops=dict(facecolor='black', arrowstyle='->'))
        
        # Rotate date labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(PLOT_PATH, dpi=300, bbox_inches='tight')
        plt.close()
        
        return PLOT_PATH, f"Successfully visualized {ticker} data"
    except Exception as e:
        logger.error(f"Error plotting data: {e}")
        return None, f"Error creating visualization: {e}"

# Predict stock price
def predict_stock_price(ticker):
    # Check if model is loaded
    model, scaler, model_status = load_lstm_model_and_scaler()
    if model is None or scaler is None:
        logger.info("Model not loaded, attempting to initialize...")
        model, scaler, model_status = initialize_model(ticker)
        if model is None:
            return "Model could not be initialized: " + model_status, ""
    
    # Fetch data for prediction (last 90 days)
    stock_data, error = get_fmp_historical_prices(ticker, limit=90)
    
    if error:
        return f"Error fetching data: {error}", ""
    
    if stock_data is None or stock_data.empty:
        return "No stock data available for prediction.", ""
    
    try:
        # Get close prices and handle missing values
        close_prices = stock_data["close"].values
        close_prices = close_prices[~np.isnan(close_prices)]
        
        if len(close_prices) < 60:
            # If we don't have enough data, use what we have
            seq_length = max(5, len(close_prices) // 3)
            logger.warning(f"Not enough data points (have {len(close_prices)}, need 60). Using sequence length {seq_length}")
        else:
            seq_length = 60
        
        # Prepare data for prediction
        scaled_data = scaler.transform(close_prices.reshape(-1, 1))
        
        # Take the last 'seq_length' points or pad if needed
        if len(scaled_data) < seq_length:
            # Pad with the first value if we don't have enough data
            padding = np.full((seq_length - len(scaled_data), 1), scaled_data[0][0])
            X = np.concatenate([padding, scaled_data])
        else:
            X = scaled_data[-seq_length:]
            
        X = np.array([X.reshape(-1)])
        X = X.reshape((1, seq_length, 1))
        
        # Make prediction
        prediction = model.predict(X)
        predicted_price_usd = float(scaler.inverse_transform(prediction)[0][0])
        
        # Get current price for comparison
        current_price = close_prices[-1]
        change = ((predicted_price_usd - current_price) / current_price) * 100
        
        # Get USD to INR conversion
        usd_inr_rate = get_usd_inr_rate()
        predicted_price_inr = predicted_price_usd * usd_inr_rate
        
        # Format prediction results
        prediction_text = (
            f"Current Price: ${current_price:.2f}\n"
            f"Predicted Price: ${predicted_price_usd:.2f} ({'+' if change >= 0 else ''}{change:.2f}%)\n"
            f"Prediction Date: {datetime.datetime.now().date() + datetime.timedelta(days=1)}"
        )
        
        inr_text = f"Predicted Price (INR): ₹{predicted_price_inr:.2f}" if predicted_price_inr else "INR conversion unavailable"
        
        return prediction_text, inr_text
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return f"Prediction error: {str(e)}", ""

# Gradio UI with improved styling
def create_app():
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 📈 Advanced Stock Price Prediction & Visualization
            
            Enter a stock ticker symbol (like AAPL, MSFT, GOOGL) and use the buttons below to visualize recent data or predict future prices.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                ticker_input = gr.Textbox(
                    label="Stock Ticker Symbol",
                    placeholder="AAPL",
                    value="AAPL",
                    info="Enter stock ticker (e.g., AAPL for Apple, MSFT for Microsoft)"
                )
                
                with gr.Row():
                    visualize_button = gr.Button("📊 Visualize Stock Data", variant="primary")
                    predict_button = gr.Button("🔮 Predict Price", variant="primary")
            
            with gr.Column(scale=1):
                status_output = gr.Textbox(label="Status", value="Ready to analyze stocks")
        
        with gr.Tabs():
            with gr.TabItem("Visualization"):
                with gr.Row():
                    stock_plot = gr.Image(label="Stock Price Chart")
            
            with gr.TabItem("Prediction"):
                with gr.Row():
                    with gr.Column():
                        prediction_output_usd = gr.Textbox(label="Price Prediction")
                        prediction_output_inr = gr.Textbox(label="Price in Indian Rupees (INR)")
        
        # Event handlers
        visualize_button.click(
            fn=lambda ticker: plot_stock_data(ticker),
            inputs=ticker_input,
            outputs=[stock_plot, status_output]
        )
        
        predict_button.click(
            fn=lambda ticker: predict_stock_price(ticker),
            inputs=ticker_input,
            outputs=[prediction_output_usd, prediction_output_inr]
        )
        
        # Initialize model on startup (in background)
        initialize_model("AAPL")
        
        gr.Markdown(
            """
            ### 📝 Notes
            - This app uses the Financial Modeling Prep API to obtain stock data
            - Predictions are based on historical data and should not be used for financial decisions
            - The LSTM model is trained on 2 years of historical data
            """
        )
    
    return app

# Initialize model (first attempt)
model, scaler, status = initialize_model()
logger.info(f"Initial model status: {status}")

# Create and launch the app
app = create_app()
app.launch()