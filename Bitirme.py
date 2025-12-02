# -----------------------------------------------------------------
# DOSYA ADI: Bitirme.py
# (Maksimum Tarihçe - 2005'ten Günümüze)
# -----------------------------------------------------------------
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

print("Sprint 1: Geniş Tarihli Global Veri Seti İndiriliyor...")

# 1. MAKRO GÖSTERGELER
macro_tickers = ["XU100.IS", "USDTRY=X", "^VIX", "TNX"]

# 2. GLOBAL PORTFÖY
stock_tickers = [
    "^GSPC", "^NDX", "XU100.IS", "^GDAXI", "^N225", "000001.SS",  # Endeksler
    "GC=F", "SI=F", "BZ=F",  # Emtialar
    "BTC-USD", "ETH-USD", "XRP-USD"  # Kripto
]

all_tickers = list(set(macro_tickers + stock_tickers))

# Uzun vade analizi için veriyi olabildiğince geriden alıyoruz
start_date = "2010-01-01"
end_date = pd.to_datetime("today").strftime("%Y-%m-%d")

try:
    print(f"Veriler çekiliyor ({start_date} - Bugün)...")
    data = yf.download(all_tickers, start=start_date, end=end_date)['Close']

    # Eksikleri doldur
    data = data.ffill().bfill()

    data.to_csv("dyd_proje_veriseti.csv")
    print("✅ Veri seti başarıyla kaydedildi.")

except Exception as e:
    print(f"❌ Hata: {e}")
    exit()

# -----------------------------------------------------------------
# REJİM TESPİTİ
# -----------------------------------------------------------------
print("Piyasa Rejimi Analizi...")
try:
    data = pd.read_csv("dyd_proje_veriseti.csv", index_col='Date', parse_dates=True)
    cols = [c for c in macro_tickers if c in data.columns]

    macro_returns = np.log(data[cols] / data[cols].shift(1)).dropna()
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(macro_returns)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    regime_labels = kmeans.fit_predict(scaled_returns)

    macro_returns['Regime'] = regime_labels
    data_with_regimes = data.loc[macro_returns.index].copy()
    data_with_regimes['Regime'] = regime_labels

    data_with_regimes.to_csv("dyd_veriseti_rejimli.csv")
    print("✅ Rejim analizi tamamlandı.")

except Exception as e:
    print(f"❌ Rejim Hatası: {e}")