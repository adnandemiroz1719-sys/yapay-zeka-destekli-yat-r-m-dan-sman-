# anp_app.py
import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO

st.set_page_config(layout="wide")
st.title("🔗 ANP (Analytic Network Process) — Basit Uygulama")

# --- AYARLAR ---
st.sidebar.header("Ayarlar")
num_criteria = st.sidebar.number_input("Kriter sayısı", min_value=1, max_value=8, value=3, step=1)
num_alts = st.sidebar.number_input("Alternatif (Seçenek) sayısı", min_value=2, max_value=8, value=3, step=1)

# İsimler
st.subheader("İsimler")
criteria = []
for i in range(num_criteria):
    criteria.append(st.text_input(f"{i+1}. Kriter adı", value=f"Kriter {i+1}"))

alts = []
for i in range(num_alts):
    alts.append(st.text_input(f"{i+1}. Alternatif adı", value=f"Alternatif {i+1}"))

# --- KRİTERLER ARASI KARŞILAŞTIRMALAR (AHP tarzı) ---
st.subheader("Kriterler arası ikili karşılaştırma (AHP) — Kriter ağırlıklarını hesaplamak için")
st.markdown("Kriterler arasındaki ikili karşılaştırma matrisini 1..9 ölçeğinde doldurun. Matris karşılıklı-kuralına (a_ji = 1/a_ij) uyarak otomatik tamamlanır.")

def pairwise_matrix_input(n, prefix):
    """Kullanıcıdan üst üçgeni alıp tam reciprocal matris döndürür."""
    M = np.ones((n, n))
    for i in range(n):
        for j in range(i+1, n):
            val = st.number_input(f"{prefix} {i+1} vs {j+1}", min_value=1.0, max_value=9.0, value=1.0, step=0.1, key=f"{prefix}-{i}-{j}")
            M[i, j] = val
            M[j, i] = 1.0 / val
    return M

crit_pairwise = pairwise_matrix_input(num_criteria, "Krit. karşılaştırma")

def eigen_priority_from_matrix(M):
    # Principal eigenvector approximation (np.linalg.eig)
    vals, vecs = np.linalg.eig(M)
    max_idx = np.argmax(np.real(vals))
    principal = np.real(vecs[:, max_idx])
    principal = np.abs(principal)  # negatif gelirse düzelt
    if principal.sum() == 0:
        # geometric mean fallback
        gm = np.prod(M, axis=1) ** (1.0 / M.shape[0])
        w = gm / gm.sum()
        return w
    w = principal / principal.sum()
    return w

crit_weights = eigen_priority_from_matrix(crit_pairwise)
st.write("Kriter ağırlıkları (ilk hesap):")
st.table(pd.DataFrame(crit_weights, index=criteria, columns=["Ağırlık"]))

# --- HER KRİTER İÇİN ALTERNATİFLERİN KARŞILAŞTIRMALARI ---
st.subheader("Her kriter altında alternatiflerin karşılaştırmaları (AHP) — alternatif önceliklerini hesaplar")
st.markdown("Her kriter için alternatiflerin n×n ikili karşılaştırma matrisini doldurun (üst üçgen).")

alt_priority_per_crit = np.zeros((num_criteria, num_alts))
for k in range(num_criteria):
    st.markdown(f"**{criteria[k]}** için alternatif karşılaştırmaları")
    M = pairwise_matrix_input(num_alts, f"Alt karşılaştırma (krit {k+1})")
    w = eigen_priority_from_matrix(M)
    alt_priority_per_crit[k, :] = w
    st.write(f"{criteria[k]} alt öncelikleri:")
    st.table(pd.DataFrame(w, index=alts, columns=[f"{criteria[k]} öncelik"]))

# --- ALTERNATİF -> KRİTER ETKİ MATRİSİ (Geri besleme) ---
st.subheader("Alternatif → Kriter etki matrisi (geri besleme)")
st.markdown("Her alternatifin kriterler üzerindeki göreli etkisini 0..1 arası değerlerle girin. Sütunlar (kriterler) için normalize edilecek.")
alt_to_crit = np.zeros((num_alts, num_criteria))
for i in range(num_alts):
    cols = st.columns(num_criteria)
    st.write(f"**{alts[i]}** etkileri (kriterler üzerinde):")
    for j in range(num_criteria):
        with cols[j]:
            alt_to_crit[i, j] = st.number_input(f"{alts[i]}→{criteria[j]}", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key=f"atc-{i}-{j}")

# Normalize sütun bazında (her sütun = kriter'e etki yapan alternatiflerin toplamı 1 olacak)
# Ancak eğer toplam 0 ise eşit dağıt.
for j in range(num_criteria):
    col_sum = alt_to_crit[:, j].sum()
    if col_sum == 0:
        alt_to_crit[:, j] = 1.0 / num_alts
    else:
        alt_to_crit[:, j] = alt_to_crit[:, j] / col_sum

st.write("Alternatif→Kriter (sütun-normalize):")
st.dataframe(pd.DataFrame(alt_to_crit, index=alts, columns=criteria))

# --- (İSTEĞE BAĞLI) KRİTER → KRİTER ETKİ MATRİSİ ---
st.subheader("Opsiyonel: Kriter → Kriter etki matrisi (kriterlerin birbirlerine etkisi)")
st.markdown("Eğer kriterler birbirini etkiliyorsa girin; yoksa boş bırakın (default 0). Her sütun normalize edilecek.")
crit_to_crit = np.zeros((num_criteria, num_criteria))
for j in range(num_criteria):
    for i in range(num_criteria):
        val = st.number_input(f"{criteria[i]} ← {criteria[j]} (etki)", min_value=0.0, max_value=10.0, value= (1.0 if i==j else 0.0), step=0.1, key=f"ctc-{i}-{j}")
        crit_to_crit[i, j] = val
# normalize columns
for j in range(num_criteria):
    s = crit_to_crit[:, j].sum()
    if s == 0:
        # eğer tüm sıfırsa identity sütunu (kendi kendine tam etki) koy
        crit_to_crit[:, j] = np.eye(num_criteria)[:, j]
    else:
        crit_to_crit[:, j] = crit_to_crit[:, j] / s

st.write("Kriter→Kriter (sütun-normalize):")
st.dataframe(pd.DataFrame(crit_to_crit, index=criteria, columns=criteria))

# --- SÜPER-MATRİS OLUŞTURMA ---
st.subheader("Süper-matris oluşturma")
n = num_criteria + num_alts
# sıra: [kriterler..., alternatifler...]
supermatrix = np.zeros((n, n))

# 1) Eğer influencer bir kriterse (sütun j < num_criteria):
#    - Etkilenen kriter satır blok (0:num_criteria): crit_to_crit[:, j]
#    - Etkilenen alternatif satır blok (num_criteria:): alt_priority_per_crit[j, :]
for j in range(num_criteria):
    # kriter->kriter
    supermatrix[0:num_criteria, j] = crit_to_crit[:, j]
    # kriter->alternatif : alternatiflerin öncelikleri o kritere göre
    supermatrix[num_criteria:n, j] = alt_priority_per_crit[j, :]

# 2) Eğer influencer bir alternatifse (sütun j >= num_criteria):
#    - Etkilenen kriter satır blok: alt_to_crit[alt_index, :]
#    - Etkilenen alternatif satır blok: (burada alternatif->alternatif etkisi almıyoruz, set 0)
for alt_idx in range(num_alts):
    col_idx = num_criteria + alt_idx
    # alternatif -> kriter etkisi (kriter satırlarına)
    supermatrix[0:num_criteria, col_idx] = alt_to_crit[alt_idx, :]
    # alternatif -> alternatif: sıfır (varsayılan)
    supermatrix[num_criteria:n, col_idx] = np.zeros(num_alts)

st.write("Süper-matris (normalleşmemiş):")
sm_df = pd.DataFrame(supermatrix, index=(criteria + alts), columns=(criteria + alts))
st.dataframe(sm_df)

# sütun-normalize (column-stochastic)
col_sums = supermatrix.sum(axis=0)
# eğer bir sütun 0 ise o sütunu eşit dağıt (genellikle olmamalı)
for j in range(n):
    if col_sums[j] == 0:
        supermatrix[:, j] = 1.0 / n
    else:
        supermatrix[:, j] = supermatrix[:, j] / col_sums[j]

st.write("Sütun-normalize Süper-matris (Column-stochastic):")
st.dataframe(pd.DataFrame(supermatrix, index=(criteria + alts), columns=(criteria + alts)))

# --- LİMİT SÜPER-MATRİS (güç yöntemi ile) ---
st.subheader("Limit süper-matris (sonuçların sabitlenmesi)")
st.markdown("Süper-matrisin yüksek kuvvetini alarak (iteratif) limit matrisine yaklaşıyoruz.")

# güç yöntemi: M^k (k büyük). Matris boyutu genelde küçük; 60 iter yeterli olur.
power = st.slider("Limit iterasyon sayısı (yüksek değer daha kesin, yavaş)", min_value=10, max_value=500, value=100, step=10)
M = np.array(supermatrix)
# raise to power via repeated multiplication
limit = np.linalg.matrix_power(M, power)

st.write(f"Süper-matris^{power}: (yaklaşık limit)")
st.dataframe(pd.DataFrame(limit, index=(criteria + alts), columns=(criteria + alts)))

# Alternatiflerin nihai önceliklerini almak: her alternatif satırı toplamı ya da
# sabit vektörden (örn. limit matrisin her sütunu eşit sonuç verecektir). Genelde
# limit matrisin herhangi bir sütunu aynı sonuç vektörünü içerir — biz satır toplamlarını alabiliriz.
alt_final_scores = limit[num_criteria:n, :].sum(axis=1)  # veya ortalama alınabilir
# normalize sonuç
if alt_final_scores.sum() == 0:
    alt_final_scores = np.ones_like(alt_final_scores) / alt_final_scores.size
else:
    alt_final_scores = alt_final_scores / alt_final_scores.sum()

res_df = pd.DataFrame(alt_final_scores, index=alts, columns=["Nihai Öncelik"])
st.write("🏁 Alternatiflerin nihai öncelikleri (normalize edilmiş):")
st.table(res_df.sort_values("Nihai Öncelik", ascending=False))

# --- EXCEL İNDİRME ---
st.subheader("📥 Raporu Excel'e indir")
buffer = BytesIO()
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    pd.DataFrame(crit_pairwise, index=criteria, columns=criteria).to_excel(writer, sheet_name="KriterPairwise")
    pd.DataFrame(crit_weights, index=criteria, columns=["AHP_krit_agirlik"]).to_excel(writer, sheet_name="Kriter_AHP_Agirlik")
    # Alternatifler per kriter
    for k in range(num_criteria):
        pd.DataFrame(alt_priority_per_crit[k, :], index=alts, columns=[f"AltOncelik_{criteria[k]}"]).to_excel(writer, sheet_name=f"AltOn_{k+1}")
    pd.DataFrame(alt_to_crit, index=alts, columns=criteria).to_excel(writer, sheet_name="Alt_to_Crit")
    pd.DataFrame(crit_to_crit, index=criteria, columns=criteria).to_excel(writer, sheet_name="Crit_to_Crit")
    pd.DataFrame(supermatrix, index=(criteria+alts), columns=(criteria+alts)).to_excel(writer, sheet_name="Supermatrix")
    pd.DataFrame(limit, index=(criteria+alts), columns=(criteria+alts)).to_excel(writer, sheet_name="Limit")
    res_df.to_excel(writer, sheet_name="Final_Alt_Scores")

st.download_button(
    label="Excel Raporunu İndir",
    data=buffer.getvalue(),
    file_name="ANP_raporu.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
