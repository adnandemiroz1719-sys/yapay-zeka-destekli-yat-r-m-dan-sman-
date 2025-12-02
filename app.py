import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import warnings
import uuid
import plotly.graph_objects as go
import yfinance as yf

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------
# 1. AYARLAR VE ZENGİNLEŞTİRİLMİŞ VARLIK BİLGİLERİ
# -----------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Hassas Yatırım Asistanı Pro")

GLOBAL_ASSETS = [
    "^GSPC", "^NDX", "XU100.IS", "^GDAXI", "^N225", "000001.SS",
    "GC=F", "SI=F", "BZ=F",
    "BTC-USD", "ETH-USD", "XRP-USD"
]

ASSET_INFO = {
    "^GSPC": {
        "name": "S&P 500", "type": "Endeks", "risk_level": 4, "fund_score": 9.5, "conf": 90,
        "desc": "ABD'nin en büyük 500 şirketi.",
        "fund_report": "Dünyanın en büyük ekonomisinin amiral gemisi. Kurumsal kârlılıklar güçlü, faiz artışlarına karşı dirençli. Uzun vadeli güvenli liman."
    },
    "^NDX": {
        "name": "Nasdaq 100", "type": "Endeks", "risk_level": 6, "fund_score": 9.0, "conf": 85,
        "desc": "ABD Teknoloji devleri.",
        "fund_report": "Yapay zeka ve teknoloji devlerinin evi. Büyüme potansiyeli çok yüksek ancak faiz hassasiyeti nedeniyle volatilitesi S&P 500'den fazladır."
    },
    "XU100.IS": {
        "name": "BIST 100", "type": "Endeks", "risk_level": 7, "fund_score": 6.5, "conf": 75,
        "desc": "Türkiye hisse senedi piyasası.",
        "fund_report": "Yüksek enflasyon ortamında getiri arayışı için tercih ediliyor. Gelişmekte olan piyasa riskleri ve kur oynaklığı dikkate alınmalı."
    },
    "^GDAXI": {
        "name": "DAX 40", "type": "Endeks", "risk_level": 5, "fund_score": 8.5, "conf": 80,
        "desc": "Alman sanayi endeksi.",
        "fund_report": "Avrupa'nın sanayi motoru. İhracat odaklı şirketler ağırlıkta olduğu için küresel ticaret verilerine duyarlıdır."
    },
    "^N225": {
        "name": "Nikkei 225", "type": "Endeks", "risk_level": 5, "fund_score": 8.0, "conf": 80,
        "desc": "Japonya borsası.",
        "fund_report": "Asya'nın en likit piyasası. Japonya Merkez Bankası'nın para politikaları ve Yen'in değeri şirket kârlılıklarını doğrudan etkiler."
    },
    "000001.SS": {
        "name": "SSE Composite", "type": "Endeks", "risk_level": 6, "fund_score": 7.0, "conf": 70,
        "desc": "Çin Şanghay borsası.",
        "fund_report": "Dünyanın fabrikası. Devlet teşvikleri ile büyüme potansiyeli var ancak regülasyon riskleri ve gayrimenkul sektörü belirsizlik yaratıyor."
    },
    "GC=F": {
        "name": "Altın (Ons)", "type": "Emtia", "risk_level": 3, "fund_score": 9.5, "conf": 85,
        "desc": "Enflasyondan korunma aracı.",
        "fund_report": "Tarihsel güvenli liman. Jeopolitik risklerde, savaş durumlarında ve ekonomik belirsizliklerde portföy sigortası görevi görür."
    },
    "SI=F": {
        "name": "Gümüş (Ons)", "type": "Emtia", "risk_level": 5, "fund_score": 7.5, "conf": 75,
        "desc": "Endüstriyel değerli metal.",
        "fund_report": "Hem güvenli liman hem de sanayi metali (Güneş panelleri, EV'ler). Altına göre daha agresif hareket eder, ekonomik büyüme dönemlerinde parlar."
    },
    "BZ=F": {
        "name": "Brent Petrol", "type": "Emtia", "risk_level": 6, "fund_score": 7.0, "conf": 70,
        "desc": "Küresel enerji piyasası.",
        "fund_report": "Küresel ekonominin kanı. OPEC kararları ve jeopolitik gerginlikler fiyatı anında etkiler. Enflasyonun öncü göstergesidir."
    },
    "BTC-USD": {
        "name": "Bitcoin", "type": "Kripto", "risk_level": 9, "fund_score": 6.5, "conf": 60,
        "desc": "Merkeziyetsiz dijital varlık.",
        "fund_report": "Dijital altın olarak görülüyor. Kurumsal kabul (ETF'ler) artıyor ancak regülasyon riski ve yüksek volatilitesi devam ediyor."
    },
    "ETH-USD": {
        "name": "Ethereum", "type": "Kripto", "risk_level": 9, "fund_score": 7.0, "conf": 65,
        "desc": "Akıllı kontrat platformu.",
        "fund_report": "DeFi ve NFT dünyasının işletim sistemi. Teknolojik güncelleştirmeler ve ağ aktivitesi fiyatı destekler."
    },
    "XRP-USD": {
        "name": "Ripple", "type": "Kripto", "risk_level": 10, "fund_score": 5.0, "conf": 55,
        "desc": "Ödeme sistemleri ağı.",
        "fund_report": "Bankalararası transferler için hızlı ve ucuz çözüm. Hukuki süreçler fiyat üzerinde baskı yaratabilir."
    }
}


# -----------------------------------------------------------------
# 2. TEKNİK ANALİZ MOTORU
# -----------------------------------------------------------------
class TechnicalAnalyst:
    def __init__(self, series):
        self.series = series
        self.close = series.iloc[-1]

    def get_analysis(self):
        # RSI
        delta = self.series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().abs()
        loss = loss.replace(0, 0.0001)
        rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]

        # MACD
        exp12 = self.series.ewm(span=12).mean()
        exp26 = self.series.ewm(span=26).mean()
        macd_line = (exp12 - exp26).iloc[-1]
        signal_line = (exp12 - exp26).ewm(span=9).mean().iloc[-1]

        # Trend
        sma50 = self.series.rolling(50).mean().iloc[-1]
        sma200 = self.series.rolling(200).mean().iloc[-1]

        # Bollinger Bands
        sma20 = self.series.rolling(20).mean().iloc[-1]
        std20 = self.series.rolling(20).std().iloc[-1]
        upper_bb = sma20 + (std20 * 2)
        lower_bb = sma20 - (std20 * 2)

        # Teknik Puanlama
        tech_points = 50
        signals = []

        # RSI Yorumu
        if rsi < 30:
            tech_points += 25;
            signals.append("RSI Dip (Alım Fırsatı)")
        elif rsi > 70:
            tech_points -= 15;
            signals.append("RSI Zirve (Dikkat)")
        else:
            tech_points += (rsi - 50) * 0.5

            # MACD Yorumu
        if macd_line > signal_line:
            tech_points += 20;
            signals.append("MACD Al Sinyali")
        else:
            tech_points -= 15;
            signals.append("MACD Sat Sinyali")

        # Trend Yorumu
        if pd.notna(sma50) and pd.notna(sma200):
            if sma50 > sma200:
                tech_points += 30;
                signals.append("Yükseliş Trendi")
            else:
                tech_points -= 20;
                signals.append("Düşüş Trendi")

        return {
            "Score": max(0, min(100, tech_points)),
            "Signals": signals,
            "RSI": rsi,
            "Trend": "Yükseliş" if pd.notna(sma50) and sma50 > sma200 else "Düşüş",
            "SMA50": sma50, "SMA200": sma200,
            "BB_Upper": upper_bb, "BB_Lower": lower_bb, "Close": self.close,
            "MACD_Signal": "AL" if macd_line > signal_line else "SAT"
        }


# -----------------------------------------------------------------
# 3. YÜKSEK HASSASİYETLİ PUANLAMA MOTORU
# -----------------------------------------------------------------
def calculate_match_score(ticker, user_prefs, market_data):
    info = ASSET_INFO.get(ticker, {})
    asset_risk = info['risk_level']
    asset_fund = info['fund_score']
    tech_score = market_data['tech']['Score']
    real_volatility = market_data['volatility'] * 100
    ml_return = market_data['ml_return']

    u_risk = user_prefs['risk_tolerance']
    u_tech = user_prefs['tech_importance']
    u_fund = user_prefs['fund_importance']
    u_vol = user_prefs['vol_tolerance']

    score = 0
    reasons = []

    # 1. RİSK UYUMU
    risk_diff = u_risk - asset_risk
    if abs(risk_diff) <= 3:
        score += 30
        reasons.append("✅ Risk profilinize uygun")
    elif risk_diff > 3:
        score += 20
        reasons.append("ℹ️ Güvenli bir liman tercihi")
    else:
        score -= 20
        reasons.append("⚠️ Profilinize göre biraz riskli")

    # 2. GETİRİ ETKİSİ
    if ml_return > 0.02:
        score += 25
        reasons.append(f"🚀 YZ Yüksek Getiri Bekliyor (%{ml_return * 100:.1f})")
    elif ml_return < -0.02:
        score -= 15
        reasons.append("🔻 YZ Düşüş Öngörüyor")

    # 3. TEKNİK PUAN
    if tech_score > 60:
        score += 20 * (u_tech / 5)
        reasons.append("📈 Teknik Göstergeler Pozitif")
    elif tech_score < 40:
        score -= 10
        reasons.append("📉 Teknik Göstergeler Zayıf")

    # 4. TEMEL PUAN
    if asset_fund >= 8:
        score += 15 * (u_fund / 5)
        reasons.append("🏢 Temel Olarak Çok Güçlü")

    if not reasons:
        reasons.append("Piyasa koşullarıyla uyumlu, nötr tercih.")

    return min(100, max(0, int(score))), reasons


# -----------------------------------------------------------------
# 4. GRAFİK MOTORU (MUM GRAFİĞİ - HAFTA SONU GİZLİ & NET)
# -----------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_ohlc_data_cached(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        # 5 Yıllık veri çekiyoruz (İsteğin üzerine)
        df = ticker.history(period="5y")
        return df
    except Exception as e:
        return pd.DataFrame()


def create_tech_chart(ticker_symbol, ticker_name, series_close):
    # --- 1. İNDİKATÖRLER ---
    sma50 = series_close.rolling(window=50).mean()
    sma200 = series_close.rolling(window=200).mean()

    # Bollinger
    sma20 = series_close.rolling(window=20).mean()
    std20 = series_close.rolling(window=20).std()
    upper_bb = sma20 + (std20 * 2)
    lower_bb = sma20 - (std20 * 2)

    # Ichimoku
    nine_period_high = series_close.rolling(window=9).max()
    nine_period_low = series_close.rolling(window=9).min()
    tenkan_sen = (nine_period_high + nine_period_low) / 2

    period26_high = series_close.rolling(window=26).max()
    period26_low = series_close.rolling(window=26).min()
    kijun_sen = (period26_high + period26_low) / 2

    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    period52_high = series_close.rolling(window=52).max()
    period52_low = series_close.rolling(window=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

    # --- 2. DESTEK / DİRENÇ ---
    recent_window = series_close.tail(126)
    resistance_level = recent_window.max()
    support_level = recent_window.min()
    current_price = series_close.iloc[-1]

    if current_price >= resistance_level:
        resistance_level = current_price * 1.05
        res_label = "Hedef (Fib)"
    else:
        res_label = "Direnç (6 Ay)"

    # --- 3. MUM VERİSİ ---
    ohlc_data = get_ohlc_data_cached(ticker_symbol)

    # --- 4. GRAFİK ÇİZİMİ ---
    fig = go.Figure()

    # İNDİKATÖRLER (Arka Planda)
    # Ichimoku (Belirgin ve Net)
    fig.add_trace(go.Scatter(
        x=series_close.index, y=senkou_span_a,
        mode='lines', line=dict(width=0.5, color='rgba(255, 0, 255, 0.3)'), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=series_close.index, y=senkou_span_b,
        mode='lines', name='Ichimoku Bulutu',
        fill='tonexty', fillcolor='rgba(255, 0, 255, 0.2)',
        line=dict(width=0.5, color='rgba(255, 0, 255, 0.3)'), showlegend=True, hoverinfo='skip'
    ))

    # Bollinger (Belirgin ve Net)
    fig.add_trace(go.Scatter(
        x=series_close.index, y=upper_bb,
        mode='lines', line=dict(width=1, color='rgba(0, 255, 255, 0.6)'), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=series_close.index, y=lower_bb,
        mode='lines', name='Bollinger Bantları',
        fill='tonexty', fillcolor='rgba(0, 255, 255, 0.15)',
        line=dict(width=1, color='rgba(0, 255, 255, 0.6)'), showlegend=True, hoverinfo='skip'
    ))

    # SMA (Kalın ve Düz Çizgiler)
    fig.add_trace(go.Scatter(
        x=series_close.index, y=sma50,
        mode='lines', name='SMA 50',
        line=dict(color='yellow', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=series_close.index, y=sma200,
        mode='lines', name='SMA 200',
        line=dict(color='orange', width=2)
    ))

    # --- FİYAT GRAFİĞİ (MUM) ---
    if not ohlc_data.empty:
        fig.add_trace(go.Candlestick(
            x=ohlc_data.index,
            open=ohlc_data['Open'],
            high=ohlc_data['High'],
            low=ohlc_data['Low'],
            close=ohlc_data['Close'],
            name='OHLC',
            increasing_line_color='#26a69a',  # Profesyonel Yeşil
            decreasing_line_color='#ef5350',  # Profesyonel Kırmızı
            line_width=1
        ))
    else:
        fig.add_trace(go.Scatter(
            x=series_close.index, y=series_close.values,
            mode='lines', name='Fiyat (Kapanış)',
            line=dict(color='white', width=2)
        ))

    # Destek / Direnç
    fig.add_hline(y=resistance_level, line_dash="dash", line_color="#00FF00",
                  annotation_text=f"{res_label}: {resistance_level:.2f}", annotation_position="top right")
    fig.add_hline(y=support_level, line_dash="dash", line_color="orange",
                  annotation_text=f"Destek (6 Ay): {support_level:.2f}", annotation_position="bottom right")

    # --- AYARLAR (Hafta Sonu Gizleme) ---
    is_crypto = "USD" in ticker_symbol and ("BTC" in ticker_symbol or "ETH" in ticker_symbol or "XRP" in ticker_symbol)
    rangebreaks_config = [] if is_crypto else [dict(bounds=["sat", "mon"])]

    fig.update_layout(
        title=f"{ticker_name} - Profesyonel Teknik Görünüm",
        yaxis_title="Fiyat",
        template="plotly_dark",
        height=600,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=80, b=20),
        xaxis=dict(
            rangebreaks=rangebreaks_config,  # HAFTA SONU BURADA GİZLENİYOR
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1 Ay", step="month", stepmode="backward"),
                    dict(count=3, label="3 Ay", step="month", stepmode="backward"),
                    dict(count=6, label="6 Ay", step="month", stepmode="backward"),
                    dict(count=1, label="1 Yıl", step="year", stepmode="backward"),
                    dict(step="all", label="Tümü (5 Yıl)")
                ]),
                bgcolor="#202020",
                activecolor="#404040"
            ),
            type="date"
        )
    )

    # Odaklanma: Son 3 AY (Mumların en net olduğu yer)
    if len(series_close) > 0:
        three_months_ago = series_close.index[-1] - pd.DateOffset(months=3)
        fig.update_xaxes(range=[three_months_ago, series_close.index[-1]])

    return fig


# -----------------------------------------------------------------
# 5. VERİ MOTORU (CANLI & YEDEKLEME)
# -----------------------------------------------------------------
def fetch_fresh_data():
    tickers = GLOBAL_ASSETS + ["^VIX", "USDTRY=X"]
    try:
        df = yf.download(tickers, period="5y", progress=False, auto_adjust=True)['Close']
        df = df.ffill().bfill()
        df.to_csv("dyd_proje_veriseti.csv")
        return df, "Veriler Başarıyla Güncellendi!"
    except Exception as e:
        return None, f"Hata: {e}"


def load_data_engine():
    with st.sidebar:
        st.divider()
        st.write("🔄 **Veri Durumu**")
        if st.button("Verileri Şimdi Güncelle (Canlı)"):
            with st.spinner("Son 5 yılın verileri çekiliyor..."):
                new_data, msg = fetch_fresh_data()
                if new_data is not None:
                    st.success(msg)
                    return new_data, "Normal/Boğa"
                else:
                    st.error(msg)

    if os.path.exists("dyd_proje_veriseti.csv"):
        try:
            raw = pd.read_csv("dyd_proje_veriseti.csv", index_col=0, parse_dates=True)
            if raw.index.tz: raw.index = raw.index.tz_localize(None)

            status = "Normal/Boğa"
            if "^VIX" in raw.columns:
                last_vix = raw['^VIX'].iloc[-1]
                if last_vix > 25:
                    status = "Stresli/Kriz"
                elif last_vix > 17:
                    status = "Dalgalı"

            return raw, status
        except:
            pass

    data, msg = fetch_fresh_data()
    return data, "Normal/Boğa"


# -----------------------------------------------------------------
# 6. ARAYÜZ VE ANA AKIŞ
# -----------------------------------------------------------------
data, status = load_data_engine()

st.title("🧠 Hassas Yatırımcı Analiz Modülü Pro")
st.markdown("*(Risk, Teknik ve Temel tercihlerinizi detaylıca işler)*")

with st.sidebar:
    st.header("👤 Profil Ayarları")
    vade = st.selectbox("Vade", ["Kısa (<1 Yıl)", "Orta (1-3 Yıl)", "Uzun (>3 Yıl)"])
    model_gun = 90 if "Kısa" in vade else (365 if "Orta" in vade else 730)

    st.divider()
    st.subheader("1. Risk Tercihi")
    risk_tolerance = st.slider("Risk İştahınız (1: Sevmem - 10: Severim)", 1, 10, 5)

    st.subheader("2. Analiz Güveni")
    tech_imp = st.slider("Teknik Analize Önem Ver", 1, 10, 5)
    fund_imp = st.slider("Temel Analize Önem Ver", 1, 10, 5)

    st.subheader("3. Psikoloji")
    vol_tolerance = st.slider("Dalgalanmaya Tolerans", 1, 10, 5)

    user_prefs = {
        "risk_tolerance": risk_tolerance,
        "tech_importance": tech_imp,
        "fund_importance": fund_imp,
        "vol_tolerance": vol_tolerance
    }

    btn_run = st.button("Analiz Et", type="primary")

col1, col2 = st.columns([3, 1])
with col1:
    if status == "Normal/Boğa":
        st.success(f"Piyasa Modu: **{status}**")
    else:
        st.warning(f"Piyasa Modu: **{status}**")

if btn_run:
    if data is None:
        st.error("Veri seti yüklenemedi.")
    else:
        scored_assets = []
        seen_tickers = set()

        with st.spinner('Piyasa verileri taranıyor ve analiz ediliyor...'):
            for ticker in GLOBAL_ASSETS:
                if ticker in seen_tickers: continue

                try:
                    if ticker not in data.columns: continue
                    series = data[ticker].dropna()

                    if len(series) < 50: continue

                    tech_eng = TechnicalAnalyst(series)
                    tech_res = tech_eng.get_analysis()
                    vol = series.pct_change().std() * np.sqrt(252)

                    ml_ret = 0
                    mp = f"dyd_ml_modelleri/model_{ticker}_{model_gun}.joblib"
                    model_ok = False

                    if os.path.exists(mp):
                        try:
                            mdl = joblib.load(mp)
                            f = pd.DataFrame()
                            for l in [30, 90, 180, 360]: f[f'mom_{l}'] = series.pct_change(l)
                            f['vol_yearly'] = series.pct_change().rolling(252).std().shift(1)
                            last_row = f.iloc[[-1]].dropna(axis=1)
                            if not last_row.empty:
                                ml_ret = mdl.predict(last_row)[0]
                                model_ok = True
                        except:
                            pass

                    if not model_ok:
                        ml_ret = series.pct_change(30).iloc[-1]

                    margin = vol * 0.5
                    ml_range_low = ml_ret - margin
                    ml_range_high = ml_ret + margin

                    m_data = {"tech": tech_res, "volatility": vol, "ml_return": ml_ret, "status": status}
                    score, reasons = calculate_match_score(ticker, user_prefs, m_data)

                    asset_info = ASSET_INFO.get(ticker, {
                        "name": ticker, "type": "Bilinmiyor",
                        "desc": "Veri yok", "fund_report": "Detaylı rapor bulunamadı.",
                        "risk_level": 5, "fund_score": 5
                    })

                    scored_assets.append({
                        "Kod": ticker,
                        "Info": asset_info,
                        "Fiyat": series.iloc[-1],
                        "Skor": score,
                        "Neden": reasons,
                        "Teknik": tech_res,
                        "Tahmin": ml_ret,
                        "Tahmin_Min": ml_range_low,
                        "Tahmin_Max": ml_range_high,
                        "Volatilite": vol
                    })

                    seen_tickers.add(ticker)

                except Exception as e:
                    pass

        scored_assets = sorted(scored_assets, key=lambda x: x['Skor'], reverse=True)

        st.divider()
        if len(scored_assets) > 0:
            best = scored_assets[0]
            st.info(f"💡 En Uygun Varlık: **{best['Info']['name']}** (Uyum Puanı: {best['Skor']})")
        else:
            st.warning("Kriterlere uygun varlık bulunamadı.")

        for i, asset in enumerate(scored_assets):
            scr = asset['Skor']
            info = asset['Info']
            tc = asset['Teknik']
            color = "🟢" if scr >= 80 else ("🟡" if scr >= 50 else "🔴")

            tech_summary = f"Varlık şu an **{tc['Trend']}** trendinde. "
            if tc['RSI'] < 30:
                tech_summary += "RSI aşırı satımda (Fırsat). "
            elif tc['RSI'] > 70:
                tech_summary += "RSI aşırı alımda (Risk). "
            else:
                tech_summary += f"RSI {tc['RSI']:.0f} ile dengeli. "

            if tc['MACD_Signal'] == "AL":
                tech_summary += "MACD pozitif momentumu destekliyor. "
            else:
                tech_summary += "MACD negatif baskı altında. "

            with st.expander(f"{color} #{i + 1} {info['name']} ({asset['Kod']}) | Puan: {scr}"):
                tab1, tab2, tab3 = st.tabs(["📊 Yatırımcı Özeti", "📈 Teknik Grafik", "🏢 Temel & Risk"])

                # --- SEKME 1: YENİ PROFESYONEL ÖZET ---
                with tab1:
                    last_price = series.iloc[-1]
                    target_price = last_price * (1 + asset['Tahmin'])
                    stop_loss = last_price * (1 - (asset['Volatilite'] * 1.5))

                    year_window = series.tail(252)
                    year_high = year_window.max()
                    year_low = year_window.min()
                    price_pos = ((last_price - year_low) / (year_high - year_low)) * 100

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("🎯 Hedef Fiyat", f"{target_price:.2f}", f"%{asset['Tahmin'] * 100:.2f}")
                    with c2:
                        st.metric("🛡️ Stop-Loss", f"{stop_loss:.2f}", f"-{(last_price - stop_loss):.2f}",
                                  delta_color="inverse")
                    with c3:
                        st.metric("📊 Volatilite", f"%{asset['Volatilite'] * 100:.1f}", "Yıllık Risk", delta_color="off")
                    with c4:
                        dist_peak = (last_price / year_high - 1) * 100
                        st.metric("🏔️ Zirveye Uzaklık", f"%{dist_peak:.1f}", f"Zirve: {year_high:.2f}")

                    st.divider()

                    col_gauge, col_info = st.columns([1, 2])
                    with col_gauge:
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=asset['Skor'],
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "YZ Alım İştahı"},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                                'bar': {'color': "rgba(0,0,0,0)"},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, 40], 'color': '#ef5350'},
                                    {'range': [40, 60], 'color': 'gray'},
                                    {'range': [60, 100], 'color': '#26a69a'}],
                                'threshold': {
                                    'line': {'color': "white", 'width': 4},
                                    'thickness': 0.75,
                                    'value': asset['Skor']}
                            }
                        ))
                        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20),
                                                paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                        st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{asset['Kod']}_{i}")

                    with col_info:
                        st.subheader("📋 Analist Notları")
                        st.caption(
                            f"52 Haftalık Konum: %{price_pos:.0f} (Dip: {year_low:.2f} - Zirve: {year_high:.2f})")
                        st.progress(int(price_pos))
                        st.write("")
                        st.markdown("**🤖 Yapay Zeka Karar Destekleri:**")
                        for r in asset['Neden']:
                            if "✅" in r or "🚀" in r:
                                st.markdown(f"<span style='color:#4CAF50'>{r}</span>", unsafe_allow_html=True)
                            elif "⚠️" in r or "🔻" in r:
                                st.markdown(f"<span style='color:#FF5252'>{r}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<span style='color:#2196F3'>{r}</span>", unsafe_allow_html=True)

                # --- SEKME 2: GRAFİK (HATA VERMEDEN ÇALIŞIR) ---
                with tab2:
                    st.markdown("### 📝 Teknik Analiz")
                    if asset['Kod'] in data.columns:
                        chart_series = data[asset['Kod']].dropna()
                        fig = create_tech_chart(asset['Kod'], info['name'], chart_series)

                        unique_key = f"chart_{asset['Kod']}_{i}_{uuid.uuid4()}"
                        st.plotly_chart(fig, use_container_width=True, key=unique_key)

                    st.info(tech_summary)
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("RSI", f"{tc['RSI']:.1f}")
                    with c2: st.metric("Trend", tc['Trend'])
                    with c3: st.metric("MACD", tc['MACD_Signal'])

                with tab3:
                    st.write(info['fund_report'])
                    st.divider()
                    c1, c2 = st.columns(2)
                    with c1: st.metric("Temel Puan", f"{info['fund_score']}/10")
                    with c2:
                        st.slider("Risk Seviyesi", 1, 10, info['risk_level'], disabled=True,
                                  key=f"rk_{i}_{asset['Kod']}")