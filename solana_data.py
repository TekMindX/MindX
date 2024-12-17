import requests
import pandas as pd

def fetch_solana_transaction_data(rpc_url, limit=100):
    """
    Fetch recent Solana blockchain transaction data using Solana RPC.

    Parameters:
        rpc_url (str): Solana RPC endpoint URL.
        limit (int): Number of transactions to fetch.

    Returns:
        pd.DataFrame: A DataFrame with 'slot' and 'transaction_count' columns.
    """
    headers = {"Content-Type": "application/json"}

    # RPC payload to fetch recent block slots
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getBlocks",
        "params": [123456, 123456 + limit]  # Replace with actual block slot range
    }

    response = requests.post(rpc_url, json=payload, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch Solana data: {response.text}")

    slots = response.json().get('result', [])
    if not slots:
        raise ValueError("No blocks returned from Solana RPC.")

    # Simulate transaction count for each slot
    tx_counts = [slot % 10 + 50 for slot in slots]  # Mocked data: replace with real logic

    # Return data as a DataFrame
    return pd.DataFrame({'Slot': slots, 'Transaction_Count': tx_counts})

if __name__ == "__main__":
    # Example usage
    solana_rpc_url = "https://api.mainnet-beta.solana.com"  # Solana mainnet endpoint
    data = fetch_solana_transaction_data(solana_rpc_url)
    print(data.head())
