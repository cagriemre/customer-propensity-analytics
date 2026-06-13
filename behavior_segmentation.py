import pandas as pd
import numpy as np
from sqlalchemy import create_engine

TRAIN_PATH = "C:/Users/Emre/Desktop/customer_behavior_train_useridTR.csv"
TEST_PATH = "C:/Users/Emre/Desktop/customer_behavior_test_useridTR.csv"
DB_URL = "postgresql://postgres:admin123@localhost:5432/customer_propensityDB"

FEATURE_COLUMNS = [
    'sepet_ikonuna_tiklama', 'listeden_sepet_ekleme', 'detaydan_sepet_ekleme',
    'hesap_sayfasina_tiklama', 'kampanyaya_tiklama', 'favoriye_ekleme',
    'beden_secimi', 'mini_sepet_kapatma', 'iade_detay_bakma', 'giris_yapma',
    'odeme_sayfasi_gorme', 'teslimat_bilgisi_gorme', 'cihaz_mobil',
    'cihaz_pc', 'cihaz_tablet', 'geri_donen_kullanici'
]

CART_ACTIONS = ['sepet_ikonuna_tiklama', 'listeden_sepet_ekleme', 'detaydan_sepet_ekleme']
BROWSE_ACTIONS = ['favoriye_ekleme', 'beden_secimi', 'kampanyaya_tiklama', 'iade_detay_bakma', 'hesap_sayfasina_tiklama']

DROP_COLUMNS = ['beden_tablosu_gorme', 'teslimat_detay_bakma']


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    combined = pd.concat([train, test], ignore_index=True)
    combined = combined.drop(columns=[c for c in DROP_COLUMNS if c in combined.columns])
    print(f"Toplam veri: {len(combined)} satir (Train: {len(train)}, Test: {len(test)})")
    return combined


def calculate_action_conversion_rates(df):
    results = []
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        total = df[col].sum()
        if total == 0:
            continue
        purchased = df[df[col] == 1]['satin_aldi'].sum()
        not_purchased = total - purchased
        rate = purchased / total
        results.append({
            'aksiyon': col,
            'toplam_yapan': int(total),
            'satin_alan': int(purchased),
            'satin_almayan': int(not_purchased),
            'donusum_orani': round(rate, 4),
            'donusum_yuzde': round(rate * 100, 2)
        })
    result_df = pd.DataFrame(results).sort_values('donusum_orani', ascending=False).reset_index(drop=True)
    print("\n=== AKSIYON CONVERSION RATES ===")
    print(result_df.to_string(index=False))
    return result_df


def calculate_funnel(df):
    total = len(df)

    sepete_ekledi = df[
        (df['sepet_ikonuna_tiklama'] == 1) |
        (df['listeden_sepet_ekleme'] == 1) |
        (df['detaydan_sepet_ekleme'] == 1)
    ]
    n_sepet = len(sepete_ekledi)

    giris_yapti = sepete_ekledi[sepete_ekledi['giris_yapma'] == 1]
    n_giris = len(giris_yapti)

    odeme_gordu = giris_yapti[giris_yapti['odeme_sayfasi_gorme'] == 1]
    n_odeme = len(odeme_gordu)

    satin_aldi = odeme_gordu[odeme_gordu['satin_aldi'] == 1]
    n_satin = len(satin_aldi)

    stages = [
        {'asama': 'Tum Kullanicilar', 'kullanici_sayisi': total, 'oran': 100.0, 'dusus_yuzde': 0.0},
        {'asama': 'Sepete Ekleme', 'kullanici_sayisi': n_sepet, 'oran': round(n_sepet / total * 100, 2), 'dusus_yuzde': round((1 - n_sepet / total) * 100, 2)},
        {'asama': 'Giris Yapma', 'kullanici_sayisi': n_giris, 'oran': round(n_giris / total * 100, 2), 'dusus_yuzde': round((1 - n_giris / n_sepet) * 100, 2) if n_sepet > 0 else 0},
        {'asama': 'Odeme Sayfasi', 'kullanici_sayisi': n_odeme, 'oran': round(n_odeme / total * 100, 2), 'dusus_yuzde': round((1 - n_odeme / n_giris) * 100, 2) if n_giris > 0 else 0},
        {'asama': 'Satin Alma', 'kullanici_sayisi': n_satin, 'oran': round(n_satin / total * 100, 2), 'dusus_yuzde': round((1 - n_satin / n_odeme) * 100, 2) if n_odeme > 0 else 0},
    ]

    funnel_df = pd.DataFrame(stages)
    print("\n=== FUNNEL ANALIZI ===")
    print(funnel_df.to_string(index=False))
    return funnel_df


def assign_behavior_segments(df):
    has_cart = (
        (df['sepet_ikonuna_tiklama'] == 1) |
        (df['listeden_sepet_ekleme'] == 1) |
        (df['detaydan_sepet_ekleme'] == 1)
    )

    has_browse = pd.Series(False, index=df.index)
    for col in BROWSE_ACTIONS:
        if col in df.columns:
            has_browse = has_browse | (df[col] == 1)

    purchased = df['satin_aldi'] == 1

    conditions = [
        purchased,
        has_cart & ~purchased,
        has_browse & ~has_cart & ~purchased,
    ]
    choices = ['Satin Alan', 'Sepet Terk Eden', 'Arastirmaci']
    df = df.copy()
    df['davranis_segmenti'] = np.select(conditions, choices, default='Pasif Gezgin')

    segment_summary = df.groupby('davranis_segmenti').agg(
        kullanici_sayisi=('userid', 'count'),
        satin_alma_orani=('satin_aldi', 'mean')
    ).reset_index()
    segment_summary['satin_alma_orani'] = (segment_summary['satin_alma_orani'] * 100).round(2)
    segment_summary = segment_summary.sort_values('kullanici_sayisi', ascending=False).reset_index(drop=True)

    print("\n=== DAVRANIS SEGMENTLERI ===")
    print(segment_summary.to_string(index=False))
    return df, segment_summary


def calculate_device_conversion(df):
    device_data = []

    for device_col, device_name in [('cihaz_mobil', 'Mobil'), ('cihaz_pc', 'PC'), ('cihaz_tablet', 'Tablet')]:
        if device_col not in df.columns:
            continue
        users = df[df[device_col] == 1]
        total = len(users)
        purchased = users['satin_aldi'].sum()
        rate = purchased / total if total > 0 else 0
        device_data.append({
            'cihaz': device_name,
            'toplam_kullanici': int(total),
            'satin_alan': int(purchased),
            'donusum_orani': round(rate, 4),
            'donusum_yuzde': round(rate * 100, 2)
        })

    device_df = pd.DataFrame(device_data)
    print("\n=== CIHAZ BAZINDA DONUSUM ===")
    print(device_df.to_string(index=False))
    return device_df


def calculate_action_count_analysis(df):
    action_cols = [c for c in FEATURE_COLUMNS if c in df.columns and c not in ['cihaz_mobil', 'cihaz_pc', 'cihaz_tablet', 'geri_donen_kullanici']]
    df = df.copy()
    df['aksiyon_sayisi'] = df[action_cols].sum(axis=1)

    analysis = df.groupby('aksiyon_sayisi').agg(
        kullanici_sayisi=('userid', 'count'),
        satin_alma_orani=('satin_aldi', 'mean')
    ).reset_index()
    analysis['satin_alma_orani'] = (analysis['satin_alma_orani'] * 100).round(2)

    print("\n=== AKSIYON SAYISI vs SATIN ALMA ===")
    print(analysis.to_string(index=False))
    return analysis


def export_to_postgresql(conversion_df, funnel_df, segment_summary, device_df):
    engine = create_engine(DB_URL)

    conversion_df.to_sql("aksiyon_donusum_oranlari", engine, if_exists="replace", index=False)
    funnel_df.to_sql("huni_analizi", engine, if_exists="replace", index=False)
    segment_summary.to_sql("davranis_segmentleri", engine, if_exists="replace", index=False)
    device_df.to_sql("cihaz_donusum", engine, if_exists="replace", index=False)

    print("\n=== POSTGRESQL EXPORT TAMAMLANDI ===")
    print("Tablolar: aksiyon_donusum_oranlari, huni_analizi, davranis_segmentleri, cihaz_donusum")
    engine.dispose()


if __name__ == "__main__":
    df = load_data()

    conversion_df = calculate_action_conversion_rates(df)
    funnel_df = calculate_funnel(df)
    df_with_segments, segment_summary = assign_behavior_segments(df)
    device_df = calculate_device_conversion(df)
    action_count_df = calculate_action_count_analysis(df)

    export_to_postgresql(conversion_df, funnel_df, segment_summary, device_df)

    print("\n=== TAMAMLANDI ===")
