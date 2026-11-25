# 👨‍👩‍👧 Family Health Plan Recommender Pro

A Streamlit web app that recommends the best family health insurance plans using **cosine similarity** (the same similarity-based logic used in my Cinemagic AI movie recommender).

## 🌐 Live demo
[Open the live app](https://github.com/antraroy/Family-Health-Plan-Recommender-Pro/edit/main/README.md)

## ✨ Features
- Enter family details (adults, children, seniors), needs (maternity, OPD, family history) and budget
- Top 3 plans with animated match-% bars, yearly/monthly family premium, hospitals and claim ratio
- Radar chart comparing your needs with any plan
- "Why this match?" explanation showing each feature's contribution
- Plan-to-plan similarity heatmap, budget filter, CSV download
- Animated gradient UI, interactive Plotly charts

## 🛠 Tech stack
Python, Streamlit, scikit-learn (cosine similarity), pandas, NumPy, Plotly

## 🚀 Run locally
```bash
pip install -r requirements.txt
streamlit run main.py
```

## 🧠 How it works
1. Family profile and each plan are converted to 7-feature vectors
2. Cosine similarity ranks plans by closeness to the family's needs
3. Family premium is estimated from a base premium and member-wise factors
4. Plans outside the budget are filtered out of the top picks

## 🎯 Benefits
- **Families:** quickly shortlist suitable plans with clear reasons
- **Agents/insurers:** consistent, explainable plan suggestions
- **Portfolio:** shows recommender systems, feature engineering, explainability and UI skills

## 🔮 Future improvements
- Use real insurer plan data (CSV/API) and hospital network search
- Add collaborative filtering from past customer choices
- Save user profiles, add login, deploy with a database
- Combine with the Premium Estimator for an end-to-end insurance advisor

> **Disclaimer:** plan data is illustrative, not an actual insurer offer.
