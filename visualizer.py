import matplotlib.pyplot as plt

def plot_solana_predictions(data, future_slots, predictions):
    """
    Plot actual Solana transaction counts and AI predictions.

    Parameters:
        data (pd.DataFrame): Actual transaction data.
        future_slots (np.ndarray): Future block slots.
        predictions (np.ndarray): Predicted transaction counts.
    """
    plt.figure(figsize=(10, 6))

    # Plot actual data
    plt.scatter(data['Slot'], data['Transaction_Count'], color='blue', label='Actual Data')

    # Plot predictions
    plt.scatter(future_slots, predictions, color='red', label='Predicted Data')

    plt.xlabel('Slot')
    plt.ylabel('Transaction Count')
    plt.title('AI-Powered Prediction of Solana Transactions')
    plt.legend()
    plt.show()
