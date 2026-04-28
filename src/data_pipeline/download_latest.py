import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_pipeline.yahoo_data import YahooData

def download_all(symbols=["VNM", "VCB", "HPG", "FPT"]):
    provider = YahooData()
    os.makedirs("data", exist_ok=True)
    
    success = True
    for sym in symbols:
        print(f"Downloading data for {sym}...")
        df = provider.get_historical_data(sym, days=1000)
        if df is not None:
            save_path = f"data/{sym}.csv"
            df.to_csv(save_path)
            print(f"Saved {len(df)} rows to {save_path}")
        else:
            print(f"Failed to fetch data for {sym}")
            success = False
            
    if not success:
        print("Some downloads failed. Exiting with error.")
        sys.exit(1)
        
if __name__ == "__main__":
    download_all()
