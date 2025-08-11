# streamlit_app.py
"""
Kullanıcı, elindeki 4 farklı model ailesinden (farklı özellik setleriyle eğitilmiş)
uygun olanı seçerek veya otomatik seçimle tahmin alabilsin.
"""
import os
import json
import joblib
import numpy as np
import streamlit as st
from dataclasses import dataclass
from typing import List, Union, Optional

@dataclass
class EnsembleModel:
    models: List
    names: List[str]
    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = [m.predict(X) for m in self.models]
        return np.mean(np.column_stack(preds), axis=1)

def _load_any(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def load_model(process: str, channel: int, models_dir: str) -> Union[EnsembleModel, object]:
    base = f"{process}_{channel}"
    ens_path = os.path.join(models_dir, f"{base}_Ensemble.pkl")
    single_path = os.path.join(models_dir, f"{base}.pkl")
    obj = _load_any(ens_path)
    if obj is not None:
        return obj
    obj = _load_any(single_path)
    if obj is not None:
        return obj
    reg_path = os.path.join(models_dir, "registry.json")
    if os.path.exists(reg_path):
        with open(reg_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
        key = f"{process}_{channel}"
        if key in reg and reg[key].get("models"):
            names = reg[key]["models"]
            models = []
            for nm in names:
                p = os.path.join(models_dir, f"{base}_{nm}.pkl")
                m = _load_any(p)
                if m is not None:
                    models.append(m)
            if models:
                return EnsembleModel(models=models, names=names[:len(models)])
    raise FileNotFoundError(f"Model bulunamadı: process={process} channel={channel} dir={models_dir}")

st.set_page_config(page_title="QCD Tesir Kesiti Tahmin Aracı", layout="centered")
st.title("QCD Tesir Kesiti Tahmin Aracı")
st.caption("process + channel bazlı, farklı özellik setleriyle eğitilmiş modellerden tesir kesiti tahmin")

paths = {
    "just_energy": os.path.join("just_energy", "models"),
    "energy_cval": os.path.join("energy+centralvalue", "models"),
    "energy_cval_abs": os.path.join("energy+centralvalues+absolute_uncertainty", "models"),
    "energy_cval_est": os.path.join("energy+centralvalues+estimated_numerical", "models"),
}

col1, col2 = st.columns(2)
with col1:
    process = st.selectbox("Process", ["LO", "NLO", "NNLO"], index=0)
with col2:
    channel = st.selectbox("Channel", [91, 92, 93, 96, 97, 98], index=0)

energy = st.number_input("energy (TeV)", value=100.0, min_value=0.0, step=1.0)

with st.expander("Opsiyonel alanlar"):
    central_value = st.text_input("Central_Value (boş bırakılabilir)", value="")
    abs_pdf = st.text_input("Absolute_PDF_Uncertainty (boş bırakılabilir)", value="")
    est_num = st.text_input("Estimated_Numerical_Uncertainty_Percent (boş bırakılabilir)", value="")

    if abs_pdf.strip() != "" and est_num.strip() != "":
        st.error("❗ Absolute_PDF_Uncertainty ve Estimated_Numerical_Uncertainty_Percent aynı anda girilemez. Lütfen yalnızca birini doldurun.")

    # Eğer ABS veya EST girilecekse Central_Value zorunlu
    if (abs_pdf.strip() != "" or est_num.strip() != "") and central_value.strip() == "":
        st.error("❗ Absolute/Estimated alanlarını kullanmak için Central_Value zorunludur.")

mode = st.radio("Model ailesi seçimi", ["Auto (girdiye göre)", "Manuel"], horizontal=True)
model_family = None

if mode == "Manuel":
    model_family = st.selectbox(
        "Model ailesi", (
            "just_energy",
            "energy_cval",
            "energy_cval_abs",
            "energy_cval_est",
        ),
        index=0,
    )
    st.markdown(":information_source: Seçilen model ailesi: **%s**" % model_family.replace("_", " "))
else:
    has_c = central_value.strip() != ""
    has_abs = abs_pdf.strip() != ""
    has_est = est_num.strip() != ""

    if (has_abs or has_est) and not has_c:
        model_family = None  # Central_Value olmadan ABS/EST kullanılamaz
    elif has_abs and has_est:
        model_family = None  # iki alan birden dolu ise otomatik seçim yapılmaz
    elif not has_c:
        model_family = "just_energy"
    else:
        if has_abs:
            model_family = "energy_cval_abs"
        elif has_est:
            model_family = "energy_cval_est"
        else:
            model_family = "energy_cval"

if st.button("Tahmin Yap"):
    try:
        # Global doğrulama: aynı anda ABS ve EST girilemez
        if abs_pdf.strip() != "" and est_num.strip() != "":
            st.error("Absolute_PDF_Uncertainty ve Estimated_Numerical_Uncertainty_Percent aynı anda girilemez. Lütfen yalnızca birini doldurun.")
            st.stop()

        # ABS veya EST girilecekse Central_Value zorunlu
        if (abs_pdf.strip() != "" or est_num.strip() != "") and central_value.strip() == "":
            st.error("Absolute/Estimated alanlarını kullanmak için Central_Value zorunludur.")
            st.stop()

        if model_family is None:
            st.error("Model ailesi belirlenemedi. Lütfen opsiyonel alanlardan yalnızca birini doldurun veya Manuel modda aile seçin.")
            st.stop()

        if model_family == "just_energy":
            X = np.array([[float(energy)]], dtype=float)
            models_dir = paths["just_energy"]
        elif model_family == "energy_cval":
            if central_value.strip() == "":
                st.error("Central_Value gerekli (energy+centralvalue).")
                st.stop()
            X = np.array([[float(energy), float(central_value)]], dtype=float)
            models_dir = paths["energy_cval"]
        elif model_family == "energy_cval_abs":
            if central_value.strip() == "" or abs_pdf.strip() == "":
                st.error("Central_Value ve Absolute_PDF_Uncertainty gerekli.")
                st.stop()
            X = np.array([[float(energy), float(central_value), float(abs_pdf)]], dtype=float)
            models_dir = paths["energy_cval_abs"]
        elif model_family == "energy_cval_est":
            if central_value.strip() == "" or est_num.strip() == "":
                st.error("Central_Value ve Estimated_Numerical_Uncertainty_Percent gerekli.")
                st.stop()
            X = np.array([[float(energy), float(central_value), float(est_num)]], dtype=float)
            models_dir = paths["energy_cval_est"]
        else:
            st.error("Model ailesi seçilemedi.")
            st.stop()

        model = load_model(process, int(channel), models_dir=models_dir)
        y_hat = model.predict(X)
        pred = float(y_hat.ravel()[0])
        st.success("Tahmin tamam")
        st.metric(label="cross_section", value=f"{pred:.6f}")

    except FileNotFoundError as e:
        st.error(str(e))
    except ValueError as e:
        st.error(f"Değer hatası: {e}")
    except Exception as e:
        st.exception(e)
