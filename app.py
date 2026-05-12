import streamlit as st
import joblib
import numpy as np

model = joblib.load("lgbm_model.pkl")

st.set_page_config(page_title="Müşteri Satın Alma Tahmini", layout="centered")

st.title("🛒 Müşteri Satın Alma Eğilimi Tahmini")
st.markdown("Müşteri davranış bilgilerini girerek satın alma olasılığını tahmin edin.")
st.divider()

# --- Cihaz Tipi ---
st.subheader("📱 Cihaz Tipi")
cihaz_secim = st.radio("Cihaz seçin:", ["Mobil", "PC", "Tablet"], horizontal=True)
cihaz_mobil = int(cihaz_secim == "Mobil")
cihaz_pc = int(cihaz_secim == "PC")
cihaz_tablet = int(cihaz_secim == "Tablet")

st.divider()

# --- Kullanıcı Tipi ---
st.subheader("👤 Kullanıcı Tipi")
kullanici_tipi = st.radio("Kullanıcı tipi:", ["İlk Kez Gelen", "Geri Dönen Kullanıcı"], horizontal=True)
geri_donen_kullanici = int(kullanici_tipi == "Geri Dönen Kullanıcı")

st.divider()

# --- Sepet Davranışları ---
st.subheader("🛍️ Sepet Davranışları")
col1, col2, col3 = st.columns(3)
with col1:
    sepet_ikonuna_tiklama = int(st.checkbox("Sepet İkonuna Tıkladı"))
with col2:
    listeden_sepet_ekleme = int(st.checkbox("Listeden Sepete Ekledi"))
with col3:
    detaydan_sepet_ekleme = int(st.checkbox("Detaydan Sepete Ekledi"))

mini_sepet_kapatma = int(st.checkbox("Mini Sepeti Kapattı"))

st.divider()

# --- Ürün Davranışları ---
st.subheader("🔍 Ürün Davranışları")
col1, col2 = st.columns(2)
with col1:
    beden_secimi = int(st.checkbox("Beden Seçti"))
with col2:
    favoriye_ekleme = int(st.checkbox("Favorilere Ekledi"))

st.divider()

# --- Sayfa Ziyaretleri ---
st.subheader("📄 Sayfa Ziyaretleri")
col1, col2, col3 = st.columns(3)
with col1:
    hesap_sayfasina_tiklama = int(st.checkbox("Hesap Sayfasını Ziyaret Etti"))
with col2:
    kampanyaya_tiklama = int(st.checkbox("Kampanyaya Tıkladı"))
with col3:
    iade_detay_bakma = int(st.checkbox("İade Detaylarını Gördü"))

col1, col2 = st.columns(2)
with col1:
    giris_yapma = int(st.checkbox("Giriş Yaptı"))
with col2:
    odeme_sayfasi_gorme = int(st.checkbox("Ödeme Sayfasını Gördü"))

teslimat_bilgisi_gorme = int(st.checkbox("Teslimat Bilgisini Gördü"))

st.divider()

# --- Tahmin ---
if st.button("🔮 Satın Alma Olasılığını Tahmin Et", use_container_width=True):
    features = np.array([[
        sepet_ikonuna_tiklama, listeden_sepet_ekleme, detaydan_sepet_ekleme,
        hesap_sayfasina_tiklama, kampanyaya_tiklama, favoriye_ekleme,
        beden_secimi, mini_sepet_kapatma, iade_detay_bakma, giris_yapma,
        odeme_sayfasi_gorme, teslimat_bilgisi_gorme, cihaz_mobil,
        cihaz_pc, cihaz_tablet, geri_donen_kullanici
    ]])

    proba = model.predict_proba(features)[0][1]
    percentage = round(proba * 100, 2)

    st.divider()

    if proba >= 0.6:
        st.success(f"### 🟢 Yüksek Eğilim: %{percentage}")
        st.markdown("Bu kullanıcı **yüksek** satın alma eğilimi gösteriyor. Kampanya hedeflemesi için öncelikli segment.")
    elif proba >= 0.3:
        st.warning(f"### 🟡 Orta Eğilim: %{percentage}")
        st.markdown("Bu kullanıcı **orta** satın alma eğilimi gösteriyor. Retargeting ile dönüşüm artırılabilir.")
    else:
        st.error(f"### 🔴 Düşük Eğilim: %{percentage}")
        st.markdown("Bu kullanıcı **düşük** satın alma eğilimi gösteriyor.")