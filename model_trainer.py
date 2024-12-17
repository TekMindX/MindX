# model_trainer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import time

def preprocess_data(df):
    """
    Preprocess the data by removing outliers and normalizing the transaction counts.
    This will help improve the performance of the AI model.
    """
    # Removing outliers (if any)
    df_cleaned = df[(df['Transaction Count'] > 0) & (df['Transaction Count'] < 10000)]

    # Normalizing the data for better model performance
    df_cleaned['Transaction Count'] = (df_cleaned['Transaction Count'] - df_cleaned['Transaction Count'].mean()) / df_cleaned['Transaction Count'].std()

    return df_cleaned

def train_linear_regression_model(df):
    """
    Train a Linear Regression model to predict transaction counts based on block numbers.
    Returns the trained model and its performance on the test data.
    """
    # Prepare features and target variable
    X = df[['Block']].values  # Features: Block number
    y = df['Transaction Count'].values  # Target: Transaction count

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Training Complete")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R2 Score: {r2}")

    return model, X_test, y_test, y_pred

def visualize_model_performance(df, X_test, y_test, y_pred, model):
    """
    Visualize the model performance with a scatter plot and regression line.
    """
    plt.figure(figsize=(10, 6))

    # Scatter plot for actual data
    plt.scatter(df['Block'], df['Transaction Count'], color='blue', label='Actual Data', alpha=0.6)

    # Plot the regression line
    plt.plot(X_test, y_pred, color='red', label='Predicted Data', linewidth=3)

    # Adding titles and labels
    plt.title(f"Linear Regression: Solana Transaction Prediction", fontsize=14)
    plt.xlabel("Block Number", fontsize=12)
    plt.ylabel("Transaction Count", fontsize=12)
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.show()

def save_trained_model(model, filename="transaction_predictor_model.pkl"):
    """
    Save the trained AI model to disk using pickle.
    This allows us to reuse the model without retraining it each time.
    """
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def load_trained_model(filename="transaction_predictor_model.pkl"):
    """
    Load a previously saved AI model from disk.
    """
    import pickle
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model
    except FileNotFoundError:
        print(f"No saved model found at {filename}. Returning None.")
        return None

def main():
    """
    Main function to fetch, preprocess, train, and visualize the Solana transaction prediction model.
    """
    # Simulate data fetching process (in a real-world scenario, you'd fetch actual data from Solana)
    data = {
        'Block': np.arange(1, 31),
        'Transaction Count': np.random.randint(1000, 5000, size=30)
    }
    df = pd.DataFrame(data)

    print("Preprocessing the data...")
    df_cleaned = preprocess_data(df)

    print("Training the Linear Regression model...")
    model, X_test, y_test, y_pred = train_linear_regression_model(df_cleaned)

    print("Visualizing model performance...")
    visualize_model_performance(df_cleaned, X_test, y_test, y_pred, model)

    print("Saving the trained model...")
    save_trained_model(model)

    # Optionally, load the model again
    # model = load_trained_model()

if __name__ == "__main__":
    main()
