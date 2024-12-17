# main.py
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
import time

# 1. Fetch Solana Blockchain Data
SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"

def fetch_solana_data():
    """
    Fetch Solana blockchain data, including slots and transaction counts.
    """
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getRecentBlockhash",
        "params": []
    }

    response = requests.post(SOLANA_RPC_URL, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Failed to fetch Solana data!")
        return None

def fetch_transaction_data():
    """
    Fetch transaction data from Solana blockchain for the past few blocks.
    """
    block_data = fetch_solana_data()
    if block_data:
        recent_blockhash = block_data['result']['value']['blockhash']
        print(f"Fetched recent blockhash: {recent_blockhash}")
        
        # Here, we can add more specific APIs to get the transaction counts per block.
        # For simplicity, let's simulate transaction counts for this example.
        transactions = np.random.randint(1000, 5000, size=30)  # Simulating transaction counts
        blocks = np.arange(1, 31)  # Blocks 1 to 30
        
        # Creating a DataFrame for transaction counts
        df = pd.DataFrame({
            "Block": blocks,
            "Transaction Count": transactions
        })

        return df
    else:
        return None

# 2. AI Model Training
def train_ai_model(df):
    """
    Train a linear regression model to predict future transaction counts.
    """
    # Prepare the data
    X = df[["Block"]].values  # Features: Block numbers
    y = df["Transaction Count"].values  # Target: Transaction counts

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Print model performance
    print(f"Model training complete. Model coefficients: {model.coef_}")
    
    return model, X_test, y_test

# 3. Predictions
def predict_transactions(model, X):
    """
    Predict transaction counts based on the trained AI model.
    """
    predictions = model.predict(X)
    return predictions

# 4. Data Visualization
def plot_predictions(df, predictions, X):
    """
    Visualize actual vs predicted transaction counts.
    """
    plt.figure(figsize=(10, 6))

    # Plot actual data
    plt.scatter(df["Block"], df["Transaction Count"], color="blue", label="Actual Data", alpha=0.6)

    # Plot predicted data
    plt.plot(X, predictions, color="red", label="Predicted Data", linewidth=3)

    # Titles and labels
    plt.title("Solana Blockchain Transaction Count Prediction", fontsize=14)
    plt.xlabel("Block Number", fontsize=12)
    plt.ylabel("Transaction Count", fontsize=12)
    plt.legend()
    
    # Show plot
    plt.grid(True)
    plt.show()

# 5. Main workflow
def main():
    print("Fetching Solana blockchain data...")
    # Fetch transaction data for the past few blocks
    df = fetch_transaction_data()
    if df is None:
        print("No data available. Exiting...")
        return
    
    print("Training AI model...")
    # Train the AI model
    model, X_test, y_test = train_ai_model(df)
    
    print("Making predictions...")
    # Predict future transaction counts
    predictions = predict_transactions(model, X_test)
    
    print("Visualizing results...")
    # Plot the predictions
    plot_predictions(df, predictions, X_test)
    
    print("Process complete!")

# 6. Scheduling or continuous execution
if __name__ == "__main__":
    while True:
        main()
        # Wait for 30 minutes before fetching the data again
        print("Waiting for next cycle...")
        time.sleep(1800)  # Sleep for 30 minutes
