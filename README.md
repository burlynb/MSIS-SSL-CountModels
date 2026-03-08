# 🦭 Steller Sea Lion Population Trend Classification
### MSIS 522 — Analytics & Machine Learning | University of Washington | Foster School of Business

An end-to-end machine learning pipeline predicting whether a Steller sea lion survey site shows a **declining population trend**, using ~50 years of NOAA/NMFS aerial survey data across Alaska and the Pacific Coast.

---

## 🌐 Live App

**[https://msis-ssl-ml-models.streamlit.app/](https://msis-ssl-ml-models.streamlit.app/)**

---

## 📁 Project Structure

```
MSIS-SSL-CountModels/
└── sea_lion_ml/          ← main project folder
    ├── app.py            ← Streamlit web application
    ├── train.py          ← Model training pipeline
    ├── data_utils.py     ← Data loading & feature engineering
    ├── requirements.txt  ← Python dependencies
    ├── README.md         ← Detailed documentation
    ├── data/
    │   └── ALLCOUNTS_v21_FED-fish.xlsx
    └── models/           ← Pre-trained model artifacts
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/burlynb/MSIS-SSL-CountModels
cd MSIS-SSL-CountModels/sea_lion_ml
pip install -r requirements.txt
python train.py          # train all models (~15 min)
streamlit run app.py     # launch the app
```

See `sea_lion_ml/README.md` for full documentation.

---

## 📊 Models Trained

| Model | F1 | AUC-ROC |
|---|---|---|
| LightGBM (GridSearchCV) | **0.988** | 0.993 |
| CART (GridSearchCV) | 0.976 | 0.992 |
| Random Forest (GridSearchCV) | 0.963 | **0.995** |
| Logistic Regression | 0.933 | 0.986 |
| LASSO | 0.920 | 0.987 |
| Ridge | 0.919 | 0.986 |
| MLP (Keras, tuned) | ~0.920 | ~0.981 |
| GAM (agTrend-inspired) | 0.916 | 0.978 |

---

## 📚 Course Info

**Course:** MSIS 522 — Analytics and Machine Learning  
**Institution:** Foster School of Business, University of Washington  
**Instructor:** Prof. Léonard Boussioux  

**Data:** NOAA National Marine Fisheries Service — Steller Sea Lion Aerial Survey Counts (used for educational purposes)
