import joblib
import pandas as pd
from sqlalchemy import create_engine

# Modeli yükle
model = joblib.load("lgbm_model.pkl")

# Feature importance al
feature_names = [
    'sepet_ikonuna_tiklama', 'listeden_sepet_ekleme', 'detaydan_sepet_ekleme',
    'hesap_sayfasina_tiklama', 'kampanyaya_tiklama', 'favoriye_ekleme',
    'beden_secimi', 'mini_sepet_kapatma', 'iade_detay_bakma', 'giris_yapma',
    'odeme_sayfasi_gorme', 'teslimat_bilgisi_gorme', 'cihaz_mobil',
    'cihaz_pc', 'cihaz_tablet', 'geri_donen_kullanici'
]

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)

# PostgreSQL'e yükle
engine = create_engine=("postgresql://postgres:admin123@localhost:5432/customer_propensityDB")
importance_df.to_sql("feature_importance", engine, if_exists="replace", index=False)

print("Feature importance yüklendi:")
print(importance_df)