# Define a function to generate stock price charts
def generate_stock_charts(price_data, symbol):
    """Generate stock price charts from time series data"""
    if "time_series" not in price_data or not price_data["time_series"]:
        print("No time series data available for generating charts")
        return None
    
    # Convert the time series data to a DataFrame
    time_series = price_data["time_series"]
    data = []
    
    for date, values in time_series.items():
        data.append({
            'date': datetime.strptime(date, '%Y-%m-%d'),
            'open': float(values['1. open']),
            'high': float(values['2. high']),
            'low': float(values['3. low']),
            'close': float(values['4. close']),
            'volume': int(values['5. volume'])
        })
    
    # Create DataFrame and sort by date
    df = pd.DataFrame(data)
    df = df.sort_values('date')
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Price subplot
    ax1.plot(df['date'], df['close'], label='Close', color='blue', linewidth=2)
    ax1.plot(df['date'], df['open'], label='Open', color='green', alpha=0.5)
    ax1.fill_between(df['date'], df['low'], df['high'], alpha=0.2, color='gray', label='High-Low Range')
    
    ax1.set_title(f'{symbol} Stock Price - Last {len(df)} Trading Days', fontsize=16)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Volume subplot
    ax2.bar(df['date'], df['volume'], color='purple', alpha=0.5, width=0.8)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis dates for volume subplot
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save the chart
    chart_filename = f"{symbol}_price_chart.png"
    plt.savefig(chart_filename)
    plt.close()
    
    print(f"Chart saved as {chart_filename}")    
    return chart_filename# Import necessary components from the Agno library

from agno.agent import Agent  # Core Agent class to create the finance agent
from agno.models.ollama import Ollama  # Ollama integration for using local LLMs
from agno.tools.calculator import CalculatorTools  # Basic calculations

# Import for Alpha Vantage API handling and environment variables
import os
import requests
import json
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import matplotlib.dates as mdates

# Load environment variables from .env file
load_dotenv()

# Alpha Vantage API information from environment variables
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
ALPHA_VANTAGE_HOST = os.getenv("ALPHA_VANTAGE_HOST", "alpha-vantage.p.rapidapi.com")

# Check if API key is available
if not ALPHA_VANTAGE_API_KEY:
    raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment variables. Please add it to your .env file.")

# Create a class for Alpha Vantage stock data
class AlphaVantageStock:
    def __init__(self):
        self.api_key = ALPHA_VANTAGE_API_KEY
        self.host = ALPHA_VANTAGE_HOST
    
    def get_stock_price(self, symbol):
        """Get daily stock price data for a given symbol"""
        url = f"https://{self.host}/query"
        
        querystring = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "compact",
            "datatype": "json"
        }
        
        headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.host
        }
        
        try:
            response = requests.get(url, headers=headers, params=querystring)
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()
            
            # Extract the time series data
            time_series = {}
            if "Time Series (Daily)" in data:
                time_series = data["Time Series (Daily)"]
                
                # Get the most recent date
                latest_date = list(time_series.keys())[0]
                latest_data = time_series[latest_date]
                
                # Create processed data with all time series
                processed_data = {
                    "symbol": symbol,
                    "latest_date": latest_date,
                    "latest": {
                        "open": latest_data["1. open"],
                        "high": latest_data["2. high"],
                        "low": latest_data["3. low"],
                        "close": latest_data["4. close"],
                        "volume": latest_data["5. volume"],
                    },
                    "time_series": time_series,
                    "meta_data": data.get("Meta Data", {})
                }
                
                return processed_data
            
            return data
        except Exception as e:
            return {"error": str(e)}

    def get_company_overview(self, symbol):
        """Get company overview data for a given symbol"""
        url = f"https://{self.host}/query"
        
        querystring = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "datatype": "json"
        }
        
        headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.host
        }
        
        try:
            response = requests.get(url, headers=headers, params=querystring)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
            
    def get_analyst_ratings(self, symbol):
        """Get analyst ratings for a given symbol"""
        url = f"https://{self.host}/query"
        
        querystring = {
            "function": "OVERVIEW",  # Analyst ratings are included in overview
            "symbol": symbol,
            "datatype": "json"
        }
        
        headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.host
        }
        
        try:
            response = requests.get(url, headers=headers, params=querystring)
            response.raise_for_status()
            data = response.json()
            
            # Extract analyst-related fields
            analyst_data = {
                "AnalystTargetPrice": data.get("AnalystTargetPrice", "N/A"),
                "AnalystRatingStrong": data.get("AnalystRatingStrong", "N/A"),
                "AnalystRatingsBuy": data.get("AnalystRatingsBuy", "N/A"),
                "AnalystRatingsHold": data.get("AnalystRatingsHold", "N/A"),
                "AnalystRatingsSell": data.get("AnalystRatingsSell", "N/A")
            }
            
            return analyst_data
        except Exception as e:
            return {"error": str(e)}

# Create an instance of AlphaVantageStock
alpha_vantage = AlphaVantageStock()

# Create a finance agent with manual stock data integration
finance_agent = Agent(
    # Identify the agent with a descriptive name
    name="Finance Agent",
    
    # Define the agent's purpose
    description="Your task is to find stock price information and company data",
    
    # Configure the agent to use locally hosted Ollama with llama3.2 model
    model=Ollama(id="llama3.2:latest"),
    
    # Only use calculator tools from Agno
    tools=[
        CalculatorTools()  # For financial calculations
    ],
    
    # Set specific behavioral instructions for the agent
    instructions=[
        "When asked about stock information, I'll fetch it and provide analysis",
        "Format financial data in readable tables",
        "Provide insightful analysis of the stock data"
    ],
    
    # Enable debug features for troubleshooting
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)

# Define a wrapper function to integrate the API with the agent
def get_stock_info(symbol):
    """Get comprehensive stock information for a given symbol"""
    # Get data from Alpha Vantage
    price_data = alpha_vantage.get_stock_price(symbol)
    overview_data = alpha_vantage.get_company_overview(symbol)
    analyst_data = alpha_vantage.get_analyst_ratings(symbol)
    
    # Extract 52-week high and low if available
    week_52_high = overview_data.get("52WeekHigh", "N/A")
    week_52_low = overview_data.get("52WeekLow", "N/A")
    dividend_per_share = overview_data.get("DividendPerShare", "N/A")
    description = overview_data.get("Description", "N/A")
    
    # Format and print the results
    print(f"\nStock data for {symbol}:")
    print(f"Latest close price: ${price_data.get('latest', {}).get('close', 'N/A')}")
    print(f"Company name: {overview_data.get('Name', 'N/A')}")
    print(f"Industry: {overview_data.get('Industry', 'N/A')}")
    print(f"Market cap: {overview_data.get('MarketCapitalization', 'N/A')}")
    print(f"52-Week Range: ${week_52_low} - ${week_52_high}")
    print(f"Dividend Per Share: ${dividend_per_share}")
    print(f"Analyst Target Price: ${analyst_data.get('AnalystTargetPrice', 'N/A')}")
    
    # Generate stock price chart
    chart_filename = generate_stock_charts(price_data, symbol)
    
    # Return enriched data to be processed by the agent
    return {
        "price_data": price_data,
        "overview_data": overview_data,
        "analyst_data": analyst_data,
        "additional_data": {
            "52WeekHigh": week_52_high,
            "52WeekLow": week_52_low,
            "DividendPerShare": dividend_per_share,
            "Description": description,
            "chart_filename": chart_filename
        }
    }

# Example usage
symbol = "MSFT"  # Can be changed to any stock symbol
print(f"Requesting comprehensive stock information for {symbol} using Alpha Vantage API...")

# Get the stock data
stock_data = get_stock_info(symbol)

# Use the agent for analysis with specific instructions to focus on requested data
finance_agent.print_response(
    f"Analyze this stock data for {symbol}: {json.dumps(stock_data, indent=2)}. "
    "Provide a comprehensive analysis including: "
    "1. Company summary and description "
    "2. Current stock price and 52-week range (high/low) "
    "3. Dividend information "
    "4. Detailed summary of analyst ratings and recommendations "
    "5. Key financial metrics "
    f"6. Mention that a price chart has been generated and saved as {stock_data.get('additional_data', {}).get('chart_filename', 'N/A')} "
    "Format the information in a clear, organized manner using tables where appropriate.",
    stream=True
)