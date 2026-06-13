import os
import sys
from typing import List, Optional
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lgbm_model.pkl")
DB_URL = "postgresql://postgres:admin123@localhost:5432/customer_propensityDB"

model = None
db_engine = None


FEATURE_ORDER = [
    'sepet_ikonuna_tiklama', 'listeden_sepet_ekleme', 'detaydan_sepet_ekleme',
    'hesap_sayfasina_tiklama', 'kampanyaya_tiklama', 'favoriye_ekleme',
    'beden_secimi', 'mini_sepet_kapatma', 'iade_detay_bakma', 'giris_yapma',
    'odeme_sayfasi_gorme', 'teslimat_bilgisi_gorme', 'cihaz_mobil',
    'cihaz_pc', 'cihaz_tablet', 'geri_donen_kullanici'
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, db_engine
    model = joblib.load(MODEL_PATH)
    db_engine = create_engine(DB_URL)
    print(f"Model yuklendi: {MODEL_PATH}")
    print(f"Veritabani baglantisi kuruldu: {DB_URL}")
    yield
    if db_engine:
        db_engine.dispose()


app = FastAPI(
    title="Customer Propensity API",
    description="Musteri satin alma egilimi tahmin API'si. LightGBM modeli ile egitilmistir.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerFeatures(BaseModel):
    sepet_ikonuna_tiklama: int = Field(0, ge=0, le=1)
    listeden_sepet_ekleme: int = Field(0, ge=0, le=1)
    detaydan_sepet_ekleme: int = Field(0, ge=0, le=1)
    hesap_sayfasina_tiklama: int = Field(0, ge=0, le=1)
    kampanyaya_tiklama: int = Field(0, ge=0, le=1)
    favoriye_ekleme: int = Field(0, ge=0, le=1)
    beden_secimi: int = Field(0, ge=0, le=1)
    mini_sepet_kapatma: int = Field(0, ge=0, le=1)
    iade_detay_bakma: int = Field(0, ge=0, le=1)
    giris_yapma: int = Field(0, ge=0, le=1)
    odeme_sayfasi_gorme: int = Field(0, ge=0, le=1)
    teslimat_bilgisi_gorme: int = Field(0, ge=0, le=1)
    cihaz_mobil: int = Field(0, ge=0, le=1)
    cihaz_pc: int = Field(0, ge=0, le=1)
    cihaz_tablet: int = Field(0, ge=0, le=1)
    geri_donen_kullanici: int = Field(0, ge=0, le=1)


class PredictionResponse(BaseModel):
    purchase_probability: float
    purchase_percentage: float
    segment: str
    segment_color: str


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_count: int
    segment_distribution: dict


def classify_segment(proba: float) -> tuple:
    if proba >= 0.6:
        return "Yuksek", "#22c55e"
    elif proba >= 0.3:
        return "Orta", "#eab308"
    else:
        return "Dusuk", "#ef4444"


def features_to_array(features: CustomerFeatures) -> np.ndarray:
    return np.array([[getattr(features, col) for col in FEATURE_ORDER]])


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "database_connected": db_engine is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(features: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model henuz yuklenmedi")

    X = features_to_array(features)
    proba = float(model.predict_proba(X)[0][1])
    segment, color = classify_segment(proba)

    return PredictionResponse(
        purchase_probability=round(proba, 4),
        purchase_percentage=round(proba * 100, 2),
        segment=segment,
        segment_color=color
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model henuz yuklenmedi")

    if len(request.customers) == 0:
        raise HTTPException(status_code=400, detail="En az 1 musteri gondermelisiniz")

    if len(request.customers) > 10000:
        raise HTTPException(status_code=400, detail="Tek seferde en fazla 10.000 musteri gonderilebilir")

    X = np.vstack([features_to_array(c) for c in request.customers])
    probas = model.predict_proba(X)[:, 1]

    predictions = []
    segment_counts = {"Yuksek": 0, "Orta": 0, "Dusuk": 0}

    for proba in probas:
        segment, color = classify_segment(float(proba))
        segment_counts[segment] += 1
        predictions.append(PredictionResponse(
            purchase_probability=round(float(proba), 4),
            purchase_percentage=round(float(proba) * 100, 2),
            segment=segment,
            segment_color=color
        ))

    return BatchPredictionResponse(
        predictions=predictions,
        total_count=len(predictions),
        segment_distribution=segment_counts
    )


@app.get("/analytics/conversion-rates")
async def get_conversion_rates():
    try:
        df = pd.read_sql("SELECT * FROM aksiyon_donusum_oranlari ORDER BY donusum_orani DESC", db_engine)
        return {"data": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/funnel")
async def get_funnel():
    try:
        df = pd.read_sql("SELECT * FROM huni_analizi", db_engine)
        return {"data": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/segments")
async def get_segments():
    try:
        df = pd.read_sql("SELECT * FROM davranis_segmentleri", db_engine)
        return {"data": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/device")
async def get_device_conversion():
    try:
        df = pd.read_sql("SELECT * FROM cihaz_donusum", db_engine)
        return {"data": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
