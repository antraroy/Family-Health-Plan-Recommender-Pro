# 🩺 Health Insurance Premium Estimator Pro

An interactive machine-learning web app (Python + Streamlit) that estimates health insurance premiums, explains **why** the premium is high or low, and projects costs over 10 years.

## ✨ Features
- **3 ML models compared** – Linear Regression, Random Forest, Gradient Boosting (R², MAE shown)
- **Explainable estimate** – shows how much each factor (smoking, BMI, exercise...) adds to the premium
- **Risk score gauge** (0–100) with animated gauge
- **What-if simulator** – quit smoking, lower BMI, change coverage and see the saving
- **Animated 10-year projection** with adjustable medical inflation (Play button)
- **Download quote** as CSV
- Animated gradient UI, tabs, interactive Plotly charts

## 🌐 Live demo
[Open the live app](PASTE-YOUR-STREAMLIT-LINK-HERE)

## 🛠 Tech stack
Python, Streamlit, scikit-learn, pandas, NumPy, Plotly

## 🚀 Run locally
```bash
pip install -r requirements.txt
streamlit run main.py
```

## 📁 Structure
```
main.py           # Complete app: data, ML models, explainability, animated UI
requirements.txt
```

## 🎯 Benefits of this project
- **For customers:** transparent premium, clear view of which habits cost money, and what-if savings.
- **For insurers/agents:** quick quote, risk scoring and consistent pricing logic.
- **For learning/portfolio:** covers the full ML workflow – data, training, model comparison, evaluation, explainability, deployment-ready UI.

## 🔧 Improvements made over a basic version
1. Single linear model → **3 models with automatic best-model selection**
2. Black-box number → **factor-wise explanation** per customer
3. Static page → **animated charts, gauges and projection**
4. Added **risk score, what-if analysis, CSV export**
5. Clean, well-organised single-file code (data → models → explainability → UI)

## 🔮 Future improvements
- Train on a real insurance dataset (e.g. Kaggle "Medical Cost Personal")
- Add SHAP for deeper explainability
- Save the trained model with `joblib` for faster startup
- Deploy free on Streamlit Community Cloud
- Add user login and quote history database

> **Disclaimer:** trained on synthetic data; estimates are illustrative, not an actual insurer quote.
