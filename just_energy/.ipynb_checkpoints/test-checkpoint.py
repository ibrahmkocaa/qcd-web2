"""
Bu betik, data_all.xlsx içindeki veriyi kullanarak process+channel gruplarına göre
sadece 'energy' girdisi ile 'cross_section' tahmini yapan modelleri eğitir ve kaydeder.

• Tekli model: doğrudan kaydedilir (Pipeline olarak)
• Çoklu model (ensemble): listedeki tüm modeller eğitilir ve ortalama tahmin alan
  bir EnsembleModel olarak kaydedilir.

Kullanılan modeller:
- Linear        -> LinearRegression
- Ridge         -> Ridge
- Poly2/3/4     -> PolynomialFeatures(degree=2/3/4) + LinearRegression

Kaydetme yapısı (varsayılan klasör: models/):
- models/LO_91.pkl                 (tekli model)
- models/LO_96_Ensemble.pkl        (ensemble model)
- (İsteğe bağlı) models/LO_96_Linear.pkl gibi tek tek modeller de kaydedilebilir.

Tahmin:
- load_model(process, channel) ile uygun modeli yükleyin
- predict(process, channel, energy) ile tahmin yapın

Not: Varsayılan veri yolu 'data_all.xlsx'. Farklı bir yol kullanıyorsanız
train_all(file_path=...) parametresini değiştirin.
"""
from __future__ import annotations
import os
import json
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# ===============================
# Kullanıcı Tanımlı Model Planı
# ===============================
# Anahtar: (process, channel)
# Değer: [model adları]
MODEL_PLAN: Dict[Tuple[str, int], List[str]] = {
    ("LO", 91): ["Ridge"],
    ("LO", 92): ["Poly3"],
    ("LO", 93): ["Ridge"],
    ("LO", 96): ["Ridge", "Poly2", "Poly3", "Poly4"],
    ("LO", 97): ["Ridge", "Poly2", "Poly3", "Poly4"],
    ("LO", 98): ["Ridge", "Poly2", "Poly3", "Poly4"],

    ("NLO", 91): ["Ridge"],
    ("NLO", 92): ["Poly3", "Poly4"],
    ("NLO", 93): ["Ridge", "Poly2", "Poly3"],
    ("NLO", 96): ["Poly2"],
    ("NLO", 97): ["Linear", "Ridge", "Poly2", "Poly3", "Poly4"],
    ("NLO", 98): ["Poly2"],

    ("NNLO", 91): ["Ridge", "Poly2", "Poly4"],
    ("NNLO", 92): ["Poly3"],
    ("NNLO", 93): ["Ridge", "Poly2", "Poly3"],
    ("NNLO", 96): ["Linear", "Ridge", "Poly2", "Poly3", "Poly4"],
    ("NNLO", 97): ["Linear", "Ridge", "Poly2", "Poly3", "Poly4"],
    ("NNLO", 98): ["Poly2"],
}

# ===============================
# Yardımcı Sınıf: Ensemble
# ===============================
@dataclass
class EnsembleModel:
    models: List[Pipeline]
    names: List[str]

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = [m.predict(X) for m in self.models]
        return np.mean(np.column_stack(preds), axis=1)

# ===============================
# Model Kurucu
# ===============================

def build_model(name: str) -> Pipeline:
    name = name.strip()
    if name == "Linear":
        return Pipeline([
            ("reg", LinearRegression())
        ])
    if name == "Ridge":
        return Pipeline([
            ("reg", Ridge(random_state=42))
        ])
    if name.startswith("Poly"):
        try:
            deg = int(name.replace("Poly", ""))
        except ValueError:
            raise ValueError(f"Geçersiz polynomial model adı: {name}")
        return Pipeline([
            ("poly", PolynomialFeatures(degree=deg, include_bias=False)),
            ("reg", LinearRegression())
        ])
    raise ValueError(f"Bilinmeyen model adı: {name}")

# ===============================
# Eğitim Fonksiyonları
# ===============================

def fit_group(X: np.ndarray, y: np.ndarray, model_names: List[str]) -> Union[Pipeline, EnsembleModel]:
    if len(model_names) == 1:
        pipe = build_model(model_names[0])
        pipe.fit(X, y)
        return pipe
    # Ensemble: tüm modelleri eğit ve ortalama al
    models = []
    for name in model_names:
        pipe = build_model(name)
        pipe.fit(X, y)
        models.append(pipe)
    return EnsembleModel(models=models, names=model_names)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ===============================
# Ana Eğitim Akışı
# ===============================

def train_all(file_path: str = "data_all.xlsx", models_dir: str = "models", save_individual: bool = True) -> None:
    """Tüm process+channel kombinasyonları için modelleri eğitir ve kaydeder."""
    ensure_dir(models_dir)

    df = pd.read_excel(file_path, sheet_name=0)

    # Sadece gerekli sütunlar
    needed = {"process", "channel", "energy", "cross_section"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Eksik sütun(lar): {missing}")

    # Model envanteri: hangi dosya kaydedildi
    registry: Dict[str, Dict[str, str]] = {}

    # Gruplar üzerinde dön
    for (process, channel), plan in MODEL_PLAN.items():
        grp = df[(df["process"] == process) & (df["channel"] == channel)]
        if grp.empty:
            print(f"Uyarı: Veri bulunamadı -> process={process}, channel={channel}")
            continue

        X = grp[["energy"]].to_numpy(dtype=float)
        y = grp["cross_section"].to_numpy(dtype=float)

        model_obj = fit_group(X, y, plan)

        # Kaydetme
        base_name = f"{process}_{channel}"
        if isinstance(model_obj, EnsembleModel):
            out_path = os.path.join(models_dir, f"{base_name}_Ensemble.pkl")
            joblib.dump(model_obj, out_path)
            print(f"Kaydedildi (ensemble): {out_path}")

            if save_individual:
                # Ensemble içindeki tek tek modelleri de kaydedelim
                for m, nm in zip(model_obj.models, model_obj.names):
                    p = os.path.join(models_dir, f"{base_name}_{nm}.pkl")
                    joblib.dump(m, p)
        else:
            out_path = os.path.join(models_dir, f"{base_name}.pkl")
            joblib.dump(model_obj, out_path)
            print(f"Kaydedildi: {out_path}")

        registry[f"{process}_{channel}"] = {
            "models": plan,
            "is_ensemble": len(plan) > 1
        }

    # Envanteri JSON olarak sakla
    reg_path = os.path.join(models_dir, "registry.json")
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    print(f"Model envanteri yazıldı: {reg_path}")


# ===============================
# Yükleme ve Tahmin
# ===============================

def _load_any(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def load_model(process: str, channel: int, models_dir: str = "models") -> Union[Pipeline, EnsembleModel]:
    base = f"{process}_{channel}"
    ens_path = os.path.join(models_dir, f"{base}_Ensemble.pkl")
    single_path = os.path.join(models_dir, f"{base}.pkl")

    obj = _load_any(ens_path)
    if obj is not None:
        return obj
    obj = _load_any(single_path)
    if obj is not None:
        return obj
    # Fallback: kayıtlı tekil modellerden ensemble oluştur (eğer varsa)
    reg_path = os.path.join(models_dir, "registry.json")
    if os.path.exists(reg_path):
        with open(reg_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
        key = f"{process}_{channel}"
        if key in reg and reg[key]["models"]:
            names = reg[key]["models"]
            models = []
            for nm in names:
                p = os.path.join(models_dir, f"{base}_{nm}.pkl")
                m = _load_any(p)
                if m is not None:
                    models.append(m)
            if models:
                return EnsembleModel(models=models, names=names[:len(models)])
    raise FileNotFoundError(f"Model bulunamadı: {process=} {channel=}")


def predict(process: str, channel: int, energy: float, models_dir: str = "models") -> float:
    model = load_model(process, channel, models_dir=models_dir)
    X = np.array([[float(energy)]], dtype=float)
    y_hat = model.predict(X)
    return float(y_hat.ravel()[0])


# ===============================
# Komut Satırı Desteği (isteğe bağlı)
# ===============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process+Channel bazlı ML eğitim ve tahmin")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Tüm modelleri eğit ve kaydet")
    p_train.add_argument("--file", default="data_all.xlsx", help="Excel dosya yolu")
    p_train.add_argument("--out", default="models", help="Model klasörü")
    p_train.add_argument("--no-save-individual", action="store_true", help="Ensemble içindeki tekli modelleri kaydetme")

    p_pred = sub.add_parser("predict", help="Tek bir tahmin yap")
    p_pred.add_argument("process", type=str, help="LO/NLO/NNLO")
    p_pred.add_argument("channel", type=int, help="91, 92, 93, 96, 97, 98 ...")
    p_pred.add_argument("energy", type=float, help="Enerji (TeV)")
    p_pred.add_argument("--models_dir", default="models", help="Model klasörü")

    args = parser.parse_args()

    if args.cmd == "train":
        train_all(file_path=args.file, models_dir=args.out, save_individual=not args.no_save_individual)
    elif args.cmd == "predict":
        y = predict(args.process, args.channel, args.energy, models_dir=args.models_dir)
        print(y)
