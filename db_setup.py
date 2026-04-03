import pandas as pd
from sqlalchemy import create_engine

engine = create_engine=("postgresql://postgres:admin123@localhost:5432/customer_propensityDB")
scores_df = pd.read_csv("customer_propensity_scores.csv")
report_df = pd.read_csv("customer_propensity_side_report.csv")

scores_df.to_sql("propensity_scores", engine, if_exists="replace",index=False)
report_df.to_sql("propensity_report",engine , if_exists="replace",index=False)

print("Tablolar oluşturuldu.")
print(f"propensity_scores: {len(scores_df)} satır")
print(f"propensity_report: {len(scores_df)} satır")