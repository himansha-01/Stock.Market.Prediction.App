Stock Price Prediction & Visualization
Overview
This project is an LSTM-based stock price prediction and visualization tool that leverages deep learning to analyze and forecast stock prices. It integrates real-time Financial Modeling Prep API data, preprocesses it using MinMaxScaler, and trains an LSTM model to make predictions. The project features an interactive Gradio UI for ease of use, along with real-time data visualization, including moving averages.

Features
📊 Fetches Real-time Stock Data using Financial Modeling Prep API
🔄 Preprocesses Data with MinMaxScaler for better model accuracy
🤖 Trains LSTM Model for stock price prediction
📈 Interactive Data Visualization with moving averages
🔮 Predicts Future Stock Prices based on historical trends
💱 Converts Prices to INR using real-time exchange rates
⚡ Optimized with Caching (LRU Cache) to minimize API calls
💾 Model Persistence using Joblib for saving and loading models
🖥️ User-Friendly Gradio UI for easy interaction
🛠️ Robust Logging & Error Handling for stability
Technologies Used
Python (TensorFlow, Keras, Scikit-learn, NumPy, Pandas)
Machine Learning (LSTM, Neural Networks)
Financial Modeling Prep API for stock market data
Matplotlib for data visualization
Gradio for UI development
Joblib for model persistence
Logging & Exception Handling for debugging
Installation & Setup
Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-repo/stock-price-prediction.git
cd stock-price-prediction
Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Run the Application
bash
Copy
Edit
python app.py
Access the UI
The application will launch in a browser via Gradio.
Enter a stock ticker (e.g., AAPL, MSFT) and click on Visualize Stock Data or Predict Price.
Usage
Enter a stock ticker symbol (e.g., AAPL, GOOGL, MSFT).
Click "Visualize Stock Data" to view recent stock trends and moving averages.
Click "Predict Price" to get a forecast for the next stock price, with USD and INR conversion.
