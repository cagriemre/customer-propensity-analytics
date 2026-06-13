import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine

st.set_page_config(
    page_title="Müşteri Satın Alma Tahmini",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_URL = "postgresql://postgres:admin123@localhost:5432/customer_propensityDB"
FEATURE_ORDER = [
    'sepet_ikonuna_tiklama', 'listeden_sepet_ekleme', 'detaydan_sepet_ekleme',
    'hesap_sayfasina_tiklama', 'kampanyaya_tiklama', 'favoriye_ekleme',
    'beden_secimi', 'mini_sepet_kapatma', 'iade_detay_bakma', 'giris_yapma',
    'odeme_sayfasi_gorme', 'teslimat_bilgisi_gorme', 'cihaz_mobil',
    'cihaz_pc', 'cihaz_tablet', 'geri_donen_kullanici'
]

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(28, 31, 38, 0.6);
        border-radius: 12px;
        padding: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
        color: #a0aec0;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-weight: 600;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(102,126,234,0.15) 0%, rgba(118,75,162,0.15) 100%);
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 12px;
        padding: 16px 20px;
    }

    div[data-testid="stMetric"] label {
        color: #a0aec0 !important;
        font-size: 0.85rem !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    .segment-card {
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin: 16px 0;
    }

    .segment-high {
        background: linear-gradient(135deg, rgba(34,197,94,0.2) 0%, rgba(16,185,129,0.1) 100%);
        border: 2px solid rgba(34,197,94,0.5);
    }

    .segment-medium {
        background: linear-gradient(135deg, rgba(234,179,8,0.2) 0%, rgba(245,158,11,0.1) 100%);
        border: 2px solid rgba(234,179,8,0.5);
    }

    .segment-low {
        background: linear-gradient(135deg, rgba(239,68,68,0.2) 0%, rgba(220,38,38,0.1) 100%);
        border: 2px solid rgba(239,68,68,0.5);
    }

    .input-section {
        background: rgba(28, 31, 38, 0.4);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
    }

    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
    }

    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.4);
    }

    .sidebar .sidebar-content {
        background: rgba(17, 19, 25, 0.95);
    }
</style>
"""


@st.cache_resource
def load_model():
    return joblib.load("lgbm_model.pkl")


@st.cache_resource
def get_db_engine():
    return create_engine(DB_URL)


@st.cache_data(ttl=300)
def load_analytics_data(table_name):
    try:
        engine = get_db_engine()
        return pd.read_sql(f"SELECT * FROM {table_name}", engine)
    except Exception:
        return None


def create_gauge_chart(value, title="Satın Alma Olasılığı"):
    if value >= 0.6:
        bar_color = "#22c55e"
    elif value >= 0.3:
        bar_color = "#eab308"
    else:
        bar_color = "#ef4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 48, "color": "white", "family": "Inter"}},
        title={"text": title, "font": {"size": 16, "color": "#a0aec0", "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4a5568", "tickfont": {"color": "#718096"}},
            "bar": {"color": bar_color, "thickness": 0.75},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(239,68,68,0.15)"},
                {"range": [30, 60], "color": "rgba(234,179,8,0.15)"},
                {"range": [60, 100], "color": "rgba(34,197,94,0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": value * 100
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"}
    )
    return fig


def classify_segment(proba):
    if proba >= 0.6:
        return "Yüksek Eğilim", "segment-high", "🟢"
    elif proba >= 0.3:
        return "Orta Eğilim", "segment-medium", "🟡"
    else:
        return "Düşük Eğilim", "segment-low", "🔴"


def render_prediction_tab():
    st.markdown("### 🎯 Müşteri Davranış Bilgilerini Girin")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("#### 📱 Cihaz Tipi")
        cihaz_secim = st.radio("Cihaz seçin:", ["Mobil", "PC", "Tablet"], horizontal=True, key="device")
        cihaz_mobil = int(cihaz_secim == "Mobil")
        cihaz_pc = int(cihaz_secim == "PC")
        cihaz_tablet = int(cihaz_secim == "Tablet")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("#### 👤 Kullanıcı Tipi")
        kullanici_tipi = st.radio("Kullanıcı tipi:", ["İlk Kez Gelen", "Geri Dönen"], horizontal=True, key="user_type")
        geri_donen_kullanici = int(kullanici_tipi == "Geri Dönen")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("#### 🛍️ Sepet Davranışları")
        c1, c2 = st.columns(2)
        with c1:
            sepet_ikonuna_tiklama = int(st.checkbox("Sepet İkonuna Tıkladı", key="cart_icon"))
            listeden_sepet_ekleme = int(st.checkbox("Listeden Sepete Ekledi", key="cart_list"))
        with c2:
            detaydan_sepet_ekleme = int(st.checkbox("Detaydan Sepete Ekledi", key="cart_detail"))
            mini_sepet_kapatma = int(st.checkbox("Mini Sepeti Kapattı", key="mini_cart"))
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("#### 🔍 Ürün Davranışları")
        c1, c2 = st.columns(2)
        with c1:
            beden_secimi = int(st.checkbox("Beden Seçti", key="size"))
        with c2:
            favoriye_ekleme = int(st.checkbox("Favorilere Ekledi", key="fav"))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("#### 📄 Sayfa Ziyaretleri")
        c1, c2 = st.columns(2)
        with c1:
            hesap_sayfasina_tiklama = int(st.checkbox("Hesap Sayfası", key="account"))
            kampanyaya_tiklama = int(st.checkbox("Kampanyaya Tıkladı", key="campaign"))
            giris_yapma = int(st.checkbox("Giriş Yaptı", key="login"))
        with c2:
            iade_detay_bakma = int(st.checkbox("İade Detayları", key="return"))
            odeme_sayfasi_gorme = int(st.checkbox("Ödeme Sayfası", key="payment"))
            teslimat_bilgisi_gorme = int(st.checkbox("Teslimat Bilgisi", key="delivery"))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")
    predict_clicked = st.button("🔮 Satın Alma Olasılığını Tahmin Et", use_container_width=True, key="predict_btn")

    if predict_clicked:
        model = load_model()
        features = np.array([[
            sepet_ikonuna_tiklama, listeden_sepet_ekleme, detaydan_sepet_ekleme,
            hesap_sayfasina_tiklama, kampanyaya_tiklama, favoriye_ekleme,
            beden_secimi, mini_sepet_kapatma, iade_detay_bakma, giris_yapma,
            odeme_sayfasi_gorme, teslimat_bilgisi_gorme, cihaz_mobil,
            cihaz_pc, cihaz_tablet, geri_donen_kullanici
        ]])

        proba = model.predict_proba(features)[0][1]
        segment_name, segment_class, segment_icon = classify_segment(proba)

        st.markdown("---")

        col1, col2 = st.columns([1, 1])
        with col1:
            fig = create_gauge_chart(proba)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(f"""
            <div class="segment-card {segment_class}">
                <div style="font-size: 3rem; margin-bottom: 8px;">{segment_icon}</div>
                <div style="font-size: 1.6rem; font-weight: 700; color: white; margin-bottom: 8px;">
                    {segment_name}
                </div>
                <div style="font-size: 2.5rem; font-weight: 800; color: white;">
                    %{round(proba * 100, 2)}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if proba >= 0.6:
                st.info("Bu kullanıcı yüksek satın alma eğilimi gösteriyor. Kampanya hedeflemesi için öncelikli segment.")
            elif proba >= 0.3:
                st.warning("Bu kullanıcı orta satın alma eğilimi gösteriyor. Yeniden hedefleme ile dönüşüm artırılabilir.")
            else:
                st.error("Bu kullanıcı düşük satın alma eğilimi gösteriyor. Yeniden etkileşim kampanyası önerilebilir.")


def render_analytics_tab():
    st.markdown("### 📊 Davranış Analizi Panosu")

    col1, col2 = st.columns(2)

    with col1:
        conversion_df = load_analytics_data("aksiyon_donusum_oranlari")
        if conversion_df is not None:
            fig = px.bar(
                conversion_df.sort_values("donusum_yuzde", ascending=True),
                x="donusum_yuzde",
                y="aksiyon",
                orientation="h",
                color="donusum_yuzde",
                color_continuous_scale=["#ef4444", "#eab308", "#22c55e"],
                labels={"donusum_yuzde": "Dönüşüm Oranı (%)", "aksiyon": "Aksiyon"}
            )
            fig.update_layout(
                title="Aksiyon Bazında Dönüşüm Oranı",
                height=500,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter", "color": "#e2e8f0"},
                showlegend=False,
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
            )
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dönüşüm oranı verisi bulunamadı. behavior_segmentation.py betiğini çalıştırın.")

    with col2:
        funnel_df = load_analytics_data("huni_analizi")
        if funnel_df is not None:
            fig = go.Figure(go.Funnel(
                y=funnel_df["asama"],
                x=funnel_df["kullanici_sayisi"],
                textinfo="value+percent initial",
                marker=dict(color=["#667eea", "#764ba2", "#f093fb", "#eab308", "#22c55e"]),
                connector=dict(line=dict(color="rgba(255,255,255,0.1)", width=1))
            ))
            fig.update_layout(
                title="Satın Alma Hunisi",
                height=500,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter", "color": "#e2e8f0"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Huni verisi bulunamadı. behavior_segmentation.py betiğini çalıştırın.")

    col3, col4 = st.columns(2)

    with col3:
        segment_df = load_analytics_data("davranis_segmentleri")
        if segment_df is not None:
            colors = {"Pasif Gezgin": "#64748b", "Araştırmacı": "#3b82f6", "Sepet Terk Eden": "#eab308", "Satın Alan": "#22c55e"}
            fig = px.pie(
                segment_df,
                values="kullanici_sayisi",
                names="davranis_segmenti",
                color="davranis_segmenti",
                color_discrete_map=colors,
                hole=0.5
            )
            fig.update_layout(
                title="Davranış Segmentleri",
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter", "color": "#e2e8f0"},
                legend=dict(font=dict(size=12))
            )
            fig.update_traces(textinfo="percent+label", textfont_size=11)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Segment verisi bulunamadı.")

    with col4:
        device_df = load_analytics_data("cihaz_donusum")
        if device_df is not None:
            colors_device = {"Mobil": "#3b82f6", "PC": "#8b5cf6", "Tablet": "#f59e0b"}
            fig = px.bar(
                device_df,
                x="cihaz",
                y="donusum_yuzde",
                color="cihaz",
                color_discrete_map=colors_device,
                text="donusum_yuzde",
                labels={"donusum_yuzde": "Dönüşüm Oranı (%)", "cihaz": "Cihaz"}
            )
            fig.update_layout(
                title="Cihaz Bazında Dönüşüm Oranı",
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter", "color": "#e2e8f0"},
                showlegend=False,
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
            )
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Cihaz verisi bulunamadı.")


def render_batch_tab():
    st.markdown("### 📁 Toplu CSV Tahmini")
    st.markdown("CSV dosyanızı yükleyin. Dosya aşağıdaki kolonları içermeli:")

    with st.expander("Gerekli Kolonlar", expanded=False):
        st.code(", ".join(FEATURE_ORDER))

    uploaded_file = st.file_uploader("CSV dosyası seçin", type=["csv"], key="batch_upload")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Dosya yüklendi: {len(df)} satır")

            missing_cols = [c for c in FEATURE_ORDER if c not in df.columns]
            if missing_cols:
                st.error(f"Eksik kolonlar: {', '.join(missing_cols)}")
                return

            model = load_model()
            X = df[FEATURE_ORDER].values
            probas = model.predict_proba(X)[:, 1]

            df["purchase_proba"] = probas.round(4)
            df["purchase_pct"] = (probas * 100).round(2)
            df["segment"] = pd.cut(
                probas,
                bins=[-1, 0.3, 0.6, 1.1],
                labels=["Düşük", "Orta", "Yüksek"]
            )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Toplam Kullanıcı", len(df))
            with col2:
                st.metric("Ort. Olasılık", f"%{df['purchase_pct'].mean():.2f}")
            with col3:
                high_count = (df["segment"] == "Yüksek").sum()
                st.metric("Yüksek Eğilim", high_count)
            with col4:
                medium_count = (df["segment"] == "Orta").sum()
                st.metric("Orta Eğilim", medium_count)

            col_chart, col_table = st.columns([1, 1])

            with col_chart:
                fig = px.histogram(
                    df,
                    x="purchase_pct",
                    nbins=50,
                    color_discrete_sequence=["#667eea"],
                    labels={"purchase_pct": "Satın Alma Olasılığı (%)"}
                )
                fig.update_layout(
                    title="Olasılık Dağılımı",
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"family": "Inter", "color": "#e2e8f0"},
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_table:
                segment_counts = df["segment"].value_counts().reset_index()
                segment_counts.columns = ["Segment", "Sayı"]
                colors_seg = {"Düşük": "#ef4444", "Orta": "#eab308", "Yüksek": "#22c55e"}
                fig = px.pie(
                    segment_counts,
                    values="Sayı",
                    names="Segment",
                    color="Segment",
                    color_discrete_map=colors_seg,
                    hole=0.5
                )
                fig.update_layout(
                    title="Segment Dağılımı",
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"family": "Inter", "color": "#e2e8f0"}
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Tahmin Sonuçları")
            display_cols = ["purchase_proba", "purchase_pct", "segment"]
            if "userid" in df.columns:
                display_cols = ["userid"] + display_cols
            st.dataframe(
                df[display_cols].sort_values("purchase_proba", ascending=False),
                use_container_width=True,
                height=400
            )

            csv_output = df[display_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Sonuçları CSV Olarak İndir",
                data=csv_output,
                file_name="batch_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Hata: {str(e)}")


def render_model_tab():
    st.markdown("### 🧠 Model Bilgileri")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", "LightGBM")
    with col2:
        st.metric("ROC-AUC", "0.9898")
    with col3:
        st.metric("F1 Score", "0.7609")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Accuracy", "0.97")
    with col5:
        st.metric("Threshold", "0.30")
    with col6:
        st.metric("Erken Durdurma", "27. iterasyon")

    st.markdown("---")

    model = load_model()
    feature_names = FEATURE_ORDER
    importances = model.feature_importances_

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        fi_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#764ba2", "#667eea", "#f093fb"],
        labels={"Importance": "Önem Skoru", "Feature": "Özellik"}
    )
    fig.update_layout(
        title="Özellik Önemliliği (Feature Importance)",
        height=550,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": "#e2e8f0"},
        showlegend=False,
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
    )
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Model Karşılaştırması")
    comparison = pd.DataFrame({
        "Model": ["Logistic Regression", "LightGBM", "XGBoost"],
        "ROC-AUC": [0.9895, 0.9898, 0.9898],
        "F1 Score": [0.689, 0.7609, 0.7610],
        "İterasyon": ["1000 (max_iter)", "27 (erken durdurma)", "500"],
        "Seçildi": ["❌", "✅", "❌"]
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown("# 🛒 Customer Propensity Analytics")
    st.markdown("*Müşteri satın alma eğilimi tahmin ve analiz platformu*")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Tekli Tahmin",
        "📊 Davranış Analizi",
        "📁 Toplu Tahmin",
        "🧠 Model Bilgileri"
    ])

    with tab1:
        render_prediction_tab()

    with tab2:
        render_analytics_tab()

    with tab3:
        render_batch_tab()

    with tab4:
        render_model_tab()

    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #4a5568; font-size: 0.8rem;">'
        'Customer Propensity Analytics Platform | LightGBM Model | 2025'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()