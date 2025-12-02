import streamlit as st
import numpy as np
import pandas as pd

st.title("🔢 Analitik Hiyerarşi Prosesi (AHP) Hesaplayıcı")

# --- SEÇENEK VE KRİTER SAYILARI SEÇİMİ ---
st.sidebar.header("⚙️ Ayarlar")
secenek_sayisi = st.sidebar.number_input("Seçenek Sayısı", min_value=2, max_value=10, value=3, step=1)
kriter_sayisi = st.sidebar.number_input("Kriter Sayısı", min_value=2, max_value=10, value=4, step=1)

# --- SEÇENEK İSİMLERİ ---
st.subheader("📝 Seçenek İsimleri")
secenekler = []
for i in range(secenek_sayisi):
    secenek = st.text_input(f"{i+1}. Seçenek İsmi", value=f"Seçenek {i+1}")
    secenekler.append(secenek)

# --- KRİTER İSİMLERİ VE AĞIRLIKLAR ---
st.subheader("📊 Kriterler ve Ağırlıkları")
kriterler = []
agirliklar = []
for i in range(kriter_sayisi):
    col1, col2 = st.columns([2, 1])
    with col1:
        kriter = st.text_input(f"{i+1}. Kriter İsmi", value=f"Kriter {i+1}")
    with col2:
        agirlik = st.slider(f"{kriter} Ağırlığı", 0.0, 1.0, 0.25, 0.01)
    kriterler.append(kriter)
    agirliklar.append(agirlik)

# --- DEĞERLER VE AÇIKLAMALAR ---
st.subheader("🔢 Seçeneklerin Kriter Değerleri ve Açıklamaları")
degerler = np.zeros((secenek_sayisi, kriter_sayisi))
aciklamalar = [["" for _ in range(kriter_sayisi)] for _ in range(secenek_sayisi)]

for i in range(secenek_sayisi):
    st.markdown(f"### {secenekler[i]}")
    for j in range(kriter_sayisi):
        col1, col2 = st.columns([1, 2])
        with col1:
            degerler[i][j] = st.number_input(
                f"{kriterler[j]} değeri",
                min_value=0.0, value=0.0, step=0.1, key=f"{i}-{j}"
            )
        with col2:
            aciklamalar[i][j] = st.text_input(
                f"{kriterler[j]} açıklama",
                value="",
                key=f"aciklama-{i}-{j}"
            )

# --- SONUÇ HESAPLAMA ---
st.subheader("📈 Sonuçlar")
sonuc = []
for i in range(secenek_sayisi):
    toplam = 0
    for j in range(kriter_sayisi):
        toplam += degerler[i][j] * agirliklar[j]
    sonuc.append(toplam)

# --- TABLO GÖSTERİMİ ---
df = pd.DataFrame(degerler, columns=kriterler, index=secenekler)
st.write("📋 Değerler Tablosu")
st.dataframe(df)

aciklama_df = pd.DataFrame(aciklamalar, columns=kriterler, index=secenekler)
st.write("📝 Açıklamalar Tablosu")
st.dataframe(aciklama_df)

st.write("Hesaplanan Sonuçlar:", sonuc)

# --- EN İYİ SEÇENEĞİ BELİRLEME ---
max_skor = max(sonuc)
en_iyi = [secenekler[i] for i, s in enumerate(sonuc) if s == max_skor]

if len(en_iyi) == 1:
    st.success(f"🏆 En iyi seçenek: **{en_iyi[0]}**")
else:
    st.warning(f"🔄 En iyi seçenekler eşit: **{', '.join(en_iyi)}**")
