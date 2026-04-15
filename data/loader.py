import pandas as pd
from pathlib import Path

# Projenin kök dizini
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"

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
    df.columns = df.columns.str.strip()
    return df

def get_shape_info(df:pd.DataFrame)->str:
    """
    DataFrame'in şekil bilgilerini döndürür.
    
    Args:
        df (pd.DataFrame): Veri seti
    Returns:
        str: Şekil bilgisi
    """
    return df.shape[0],df.shape[1],df.isnull().columns.tolist(),df.dtypes.to_dict()