    # -----------------------------------------------------------------
    # DOSYA ADI: model_egitimi.py
    # (Kısa, Orta ve Uzun Vade Modelleri)
    # -----------------------------------------------------------------
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    import joblib
    from sklearn.model_selection import train_test_split
    import os
    import warnings

    warnings.filterwarnings("ignore")

    print("Sprint 4: Farklı Vadeler İçin Modeller Eğitiliyor...")

    try:
        data = pd.read_csv("dyd_proje_veriseti.csv", index_col='Date', parse_dates=True)
    except:
        print("HATA: csv yok.")
        exit()

    stock_tickers = [
        "^GSPC", "^NDX", "XU100.IS", "^GDAXI", "^N225", "000001.SS",
        "GC=F", "SI=F", "BZ=F",
        "BTC-USD", "ETH-USD", "XRP-USD"
    ]

    # Modellerin Tahmin Edeceği Gün Sayıları
    # 90 Gün (Kısa), 365 Gün (Orta - 1 Yıl), 730 Gün (Uzun - 2 Yıl+)
    VADE_GUNLERI = [90, 365, 730]

    models_directory = "dyd_ml_modelleri"
    if not os.path.exists(models_directory):
        os.makedirs(models_directory)

    count = 0
    total = len(stock_tickers) * len(VADE_GUNLERI)

    for ticker in stock_tickers:
        if ticker not in data.columns: continue

        print(f"\n--- Analiz: {ticker} ---")

        for vade in VADE_GUNLERI:
            count += 1
            print(f"  [{count}/{total}] Hedef: {vade} Gün Sonrası...")

            df = data[[ticker]].copy()

            # Uzun vade tahminleri için daha uzun geçmişe bakmalı (200 günlük ortalama vb.)
            for lag in [30, 90, 180, 360]:
                df[f'mom_{lag}'] = df[ticker].pct_change(lag)

            # Volatilite
            df['vol_yearly'] = df[ticker].pct_change().rolling(252).std().shift(1)

            # HEDEF
            df['target'] = df[ticker].pct_change(vade).shift(-vade)
            df = df.dropna()

            if df.empty: continue

            X = df[[c for c in df.columns if 'mom_' in c or 'vol_' in c]]
            y = df['target']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, random_state=42)
            model.fit(X_train, y_train)

            filename = f"model_{ticker}_{vade}.joblib"
            joblib.dump(model, os.path.join(models_directory, filename))

    print("\n✅ TÜM VADE MODELLERİ EĞİTİLDİ.")