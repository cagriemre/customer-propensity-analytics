import pandas as pd
from sqlalchemy import create_engine

engine = create_engine=("postgresql://postgres:admin123@localhost:5432/customer_propensityDB")
scores_df = pd.read_csv("customer_propensity_scores.csv")
report_df = pd.read_csv("customer_propensity_side_report.csv")

# Segment eşiklerini düzelt
def segment(p):
    if p >= 0.6:
        return 'Yüksek'
    elif p >= 0.3:
        return 'Orta Seviye'
    else:
        return 'Düşük'

report_df['niyet_segmenti'] = report_df['purchase_proba'].apply(segment)

# Top %5 ve %10 yeniden hesapla
yuzde_95 = report_df['purchase_proba'].quantile(0.95)
yuzde_90 = report_df['purchase_proba'].quantile(0.90)
report_df['top_yuzde5'] = (report_df['purchase_proba'] >= yuzde_95).astype(int)
report_df['top_yuzde10'] = (report_df['purchase_proba'] >= yuzde_90).astype(int)

# Sırala
report_df = report_df.sort_values('purchase_proba', ascending=False).reset_index(drop=True)

# PostgreSQL'e yükle
scores_df.to_sql("propensity_scores", engine, if_exists="replace", index=False)
report_df.to_sql("propensity_report", engine, if_exists="replace", index=False)

print("Güncellendi.")
print(report_df['niyet_segmenti'].value_counts())