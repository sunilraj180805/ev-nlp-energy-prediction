# ⚡ EV Energy Consumption Prediction via NLP Knowledge Transfer

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/BERT-Sentence--Transformers-orange?logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-red?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Notebook](https://img.shields.io/badge/Jupyter-4%20Notebooks-orange?logo=jupyter)

> Predicting EV energy consumption by combining structured sensor data with domain knowledge extracted from unstructured EV manuals using BERT — even for brand-new vehicles with zero historical driving data.

---

## 🧠 The Core Idea

Training ML models for new EVs is hard — they have **no historical driving data**. 

This project solves that using **Cross-Modal Knowledge Transfer**:

1. Train models on structured sensor data (speed, battery, temperature, etc.) from existing EVs
2. Extract domain knowledge from **EV owner manuals (PDFs)** using BERT sentence embeddings
3. Transfer that NLP-derived knowledge to **boost** the structured models
4. Apply the same pipeline to **a brand-new EV** — predicting its energy consumption purely from its manual, before it ever hits the road

---

## 🔁 Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    NOTEBOOK 1                               │
│  EV Sensor CSV → Feature Engineering → RF / XGBoost / LSTM │
└─────────────────────────┬───────────────────────────────────┘
                          │ saved models
┌─────────────────────────▼───────────────────────────────────┐
│                    NOTEBOOK 2                               │
│  EV Manuals (PDF) → PyMuPDF → NLTK → TF-IDF + BERT         │
│  → Knowledge Weights & Domain Embeddings                    │
└─────────────────────────┬───────────────────────────────────┘
                          │ NLP artifacts
┌─────────────────────────▼───────────────────────────────────┐
│                    NOTEBOOK 3                               │
│  Apply NLP weights → Boost features → Train Enhanced Models │
│  Structured-only vs NLP-Enhanced Comparison                 │
└─────────────────────────┬───────────────────────────────────┘
                          │ best model
┌─────────────────────────▼───────────────────────────────────┐
│                    NOTEBOOK 4                               │
│  New Car Manual → Extract specs → Simulate driving →        │
│  Predict energy consumption (zero historical data needed)   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
ev-nlp-energy-prediction/
│
├── notebooks/
│   ├── 01_structured_data_modeling.ipynb   # EDA, feature engineering, RF/XGB/LSTM training
│   ├── 02_nlp_knowledge_extraction.ipynb   # PDF extraction, BERT embeddings, knowledge weights
│   ├── 03_combined_model.ipynb             # NLP-enhanced training + baseline comparison
│   └── 04_new_car_prediction.ipynb         # Zero-data new EV prediction demo
│
├── data/
│   ├── EV_Energy_Consumption_Dataset.csv   # Structured EV sensor dataset (5,000 records)
│   └── unstructured/                       # EV owner manual PDFs (place yours here)
│
├── saved_models/                           # Auto-generated model artifacts (.pkl, .keras)
├── results/                                # Auto-generated charts and CSVs
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**EV Energy Consumption Dataset** — 5,000 timestamped EV driving records

| Feature | Description |
|---|---|
| `Speed_kmh` | Vehicle speed |
| `Acceleration_ms2` | Instantaneous acceleration |
| `Battery_State_%` | Battery charge level |
| `Battery_Temperature_C` | Battery thermal state |
| `Temperature_C` | Ambient temperature |
| `Wind_Speed_ms` | Wind speed |
| `Driving_Mode` | Eco / Normal / Sport |
| `Road_Type` | City / Highway / Rural |
| `Traffic_Condition` | Low / Medium / High |
| `Slope_%` | Road gradient |
| `Vehicle_Weight_kg` | Vehicle load |
| **`Energy_Consumption_kWh`** | **Target** (range: 1.88 – 14.66 kWh) |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| BERT (Sentence-Transformers) | Domain knowledge extraction from PDFs |
| TF-IDF + NLTK | Text preprocessing and keyword extraction |
| PyMuPDF (`fitz`) | PDF text extraction |
| Random Forest + XGBoost | Structured data baseline models |
| LSTM (TensorFlow/Keras) | Sequential pattern learning |
| Scikit-learn | Preprocessing, metrics, PCA |
| Pandas + NumPy | Data manipulation |
| Matplotlib + Seaborn | Visualization |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/sunilraj180805/ev-nlp-energy-prediction.git
cd ev-nlp-energy-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add EV Manuals (for Notebook 2 & 4)
Place EV owner manual PDFs inside `data/unstructured/`. Any publicly available EV manual (Tesla, BMW, Hyundai, etc.) works.

### 4. Run Notebooks in Order
```
01 → 02 → 03 → 04
```
Each notebook saves its artifacts so the next one can load them automatically.

---

## 🔬 How NLP Knowledge Transfer Works

**Step 1 — Extract domain knowledge from manuals:**
BERT (`all-MiniLM-L6-v2`) encodes energy-relevant sentences from EV PDFs into dense semantic vectors.

**Step 2 — Map knowledge to features:**
TF-IDF scores identify which physical features (speed, temperature, battery) are most discussed in manuals, producing a `knowledge_weights` dictionary.

**Step 3 — Boost structured features:**
NLP-important features are amplified in the training data using the extracted weights. PCA-compressed BERT embeddings are appended as additional features.

**Step 4 — Zero-data new car prediction:**
For a new EV with no sensor data, specs are extracted from its manual, realistic driving scenarios are simulated, and the NLP-enhanced model predicts energy consumption.

---

## ⚠️ Limitations

- Requires text-based (not scanned) PDF manuals for NLP extraction
- Simulated driving scenarios in Notebook 4 are approximations — real sensor data always produces better accuracy
- Dataset is synthetic/structured; performance on real-world telemetry may vary
- BERT embedding step requires ~500MB model download on first run

---

## 🔮 Future Scope

- Fine-tune BERT on EV-specific corpus for better domain alignment
- Extend to 3D battery degradation modeling over time
- Real-time prediction API with FastAPI
- Support for multilingual EV manuals

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋 Author

**Sunilraj D**  
[GitHub](https://github.com/sunilraj180805)
