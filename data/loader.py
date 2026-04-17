import pandas as pd
from pathlib import Path

# Projenin kök dizini
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR /"kobia-credit-scorer" /"data" / "raw"

def load_uci_credit(filename:str ="uci_credit.csv")-> pd.DataFrame:
    """
    UCI Credit Card Default veri setini yükler.
    
    Returns:
        pd.DataFrame: Ham veri seti
    Raises:
        FileNotFoundError: Dosya bulunamazsa
    """
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Dosya bulunamadi: {filepath}")
    df = pd.read_csv(filepath,header=1)
    df.columns = (df.columns
              .str.strip()
              .str.lower()
              .str.replace(r"[ .]", "_", regex=True))
    return df

def get_shape_info(df:pd.DataFrame)->dict:
    """
    DataFrame'in şekil bilgilerini döndürür.
    
    Args:
        df (pd.DataFrame): Veri seti
    Returns:
        dict: Şekil bilgisi
    """
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "null_columns": df.columns[df.isnull().any()].tolist(),
        "default_payment_next_month_counts": df["default_payment_next_month"].value_counts(normalize=True).to_dict()
    }