"""Family Health Plan Recommender Pro - complete single-file version.
Content/profile-based recommender using cosine similarity (same idea as Cinemagic AI).
Run: streamlit run main.py"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ===================== DATA =====================
FEATS = ["Cover", "Maternity", "OPD", "Critical illness", "Senior care", "Child care", "Affordability"]
# name, cover(lakh), maternity, opd, critical, senior, child, affordability (0-5), base premium (1 adult, Rs/yr), network hospitals, claim ratio %
PLANS = pd.DataFrame([
    ["Budget Floater",         3, 0, 0, 0, 1, 1, 5,  4500,  4500, 88],
    ["Family Shield Basic",    5, 0, 0, 0, 1, 2, 5,  5500,  6000, 90],
    ["Family Care Plus",      10, 2, 0, 0, 1, 2, 4,  7500,  7500, 92],
    ["Little Steps",          10, 5, 2, 0, 0, 5, 3,  8500,  8000, 93],
    ["Maternity Max",         15, 5, 3, 1, 0, 4, 3, 10500,  8500, 91],
    ["Parents Care",          15, 0, 3, 2, 5, 0, 3, 11500,  7000, 89],
    ["Senior Secure",         15, 0, 2, 3, 5, 0, 2, 13000,  7200, 90],
    ["Total Wellness",        25, 3, 5, 3, 3, 3, 2, 14500, 10000, 94],
    ["Critical Cover",        25, 0, 1, 5, 2, 1, 2, 12500,  9000, 92],
    ["Platinum Family",       50, 4, 5, 5, 4, 4, 1, 21000, 12000, 95],
    ["Super Top-up Shield",   50, 0, 0, 3, 2, 1, 3,  6500,  9500, 91],
    ["Young Family Starter",  10, 3, 1, 1, 0, 4, 4,  7000,  7800, 92],
], columns=["Plan", *FEATS, "Base premium", "Hospitals", "Claim ratio %"])
MAT = PLANS[FEATS].values.astype(float)


def scale(v):
    v = np.array(v, dtype=float).copy()
    v[..., 0] = np.minimum(v[..., 0] / 10, 5)   # cover lakh -> 0..5 so it doesn't dominate
    return v


def user_vector(cover, baby, opd, history, seniors, kids, budget_priority):
    return [cover, 5 if baby else .5, 5 if opd else .5, 5 if history else 1,
            5 if seniors else .5, 5 if kids else .5, budget_priority]


def family_premium(base, adults, kids, seniors):
    return base * (1 + .8 * (adults - 1) + .35 * kids + 1.1 * seniors)


def recommend(uvec, adults, kids, seniors, max_budget):
    sims = cosine_similarity(scale([uvec]), scale(MAT))[0]
    df = PLANS.copy()
    df["Match %"] = (sims * 100).round(1)
    df["Family premium"] = [round(family_premium(b, adults, kids, seniors)) for b in df["Base premium"]]
    df["Within budget"] = df["Family premium"] <= max_budget
    return df.sort_values("Match %", ascending=False).reset_index(drop=True)


def contributions(uvec, plan_row):
    u, p = scale(uvec), scale(plan_row[FEATS].values.astype(float))
    c = u * p / (np.linalg.norm(u) * np.linalg.norm(p))
    return pd.Series(c * 100, index=FEATS)


# ===================== UI =====================
st.set_page_config(page_title="Family Plan Recommender Pro", page_icon="👨‍👩‍👧", layout="wide")
st.markdown("""
<style>
@keyframes grad {0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes fadeUp {from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:none}}
@keyframes grow {from{width:0}}
.hero{padding:28px;border-radius:16px;color:#fff;background:linear-gradient(120deg,#be185d,#7c3aed,#2563eb);
background-size:300% 300%;animation:grad 8s ease infinite, fadeUp .8s ease;margin-bottom:18px}
.hero h1{margin:0;font-size:2rem}.hero p{margin:6px 0 0;opacity:.9}
.plan{padding:16px;border-radius:14px;border:1px solid rgba(128,128,128,.35);margin-bottom:14px;animation:fadeUp .9s ease}
.plan h3{margin:0 0 6px}
.bar{height:12px;border-radius:8px;background:rgba(128,128,128,.25);overflow:hidden;margin:8px 0}
.fill{height:100%;border-radius:8px;background:linear-gradient(90deg,#22c55e,#2563eb);animation:grow 1.4s ease}
.tag{display:inline-block;padding:2px 10px;margin:2px;border-radius:20px;background:rgba(37,99,235,.15);font-size:.8rem}
</style>
<div class="hero"><h1>👨‍👩‍👧 Family Health Plan Recommender Pro</h1>
<p>Find the best-fit family health plan using similarity-based matching, with clear reasons for every recommendation</p></div>
""", unsafe_allow_html=True)

sb = st.sidebar
sb.header("Your family")
adults = sb.number_input("Adults", 1, 4, 2)
kids = sb.number_input("Children", 0, 5, 1)
seniors = sb.number_input("Senior members (60+)", 0, 3, 0)
cover = sb.selectbox("Desired cover (₹ lakh)", [3, 5, 10, 15, 25, 50], index=2)
sb.header("Needs")
baby = sb.toggle("Planning a baby (maternity)")
opd = sb.toggle("Want OPD cover")
history = sb.toggle("Family history of serious illness")
sb.header("Budget")
max_budget = sb.slider("Max yearly premium (₹)", 5000, 120000, 40000, step=1000)
bp = sb.slider("How important is a low premium?", 1, 5, 3)

uvec = user_vector(cover, baby, opd, history, seniors, kids, bp)
res = recommend(uvec, adults, kids, seniors, max_budget)
ok = res[res["Within budget"]]
top = (ok if len(ok) else res).head(3)
if not len(ok):
    st.warning("No plan fits your budget. Showing the closest matches - consider increasing the budget.")

t1, t2, t3, t4 = st.tabs(["⭐ Recommendations", "📊 Compare", "🧠 Why this match?", "ℹ️ How it works"])

with t1:
    medals = ["🥇", "🥈", "🥉"]
    for i, (_, r) in enumerate(top.iterrows()):
        perks = [f for f in ["Maternity", "OPD", "Critical illness", "Senior care", "Child care"] if r[f] >= 3]
        tags = "".join(f'<span class="tag">{p}</span>' for p in perks) or '<span class="tag">Basic hospitalisation</span>'
        st.markdown(f"""<div class="plan"><h3>{medals[i]} {r.Plan} — {r['Match %']:.0f}% match</h3>
        <div class="bar"><div class="fill" style="width:{r['Match %']}%"></div></div>
        Cover <b>₹{int(r.Cover)} lakh</b> &nbsp;|&nbsp; Family premium <b>₹{int(r['Family premium']):,}/yr</b>
        (₹{int(r['Family premium'] / 12):,}/month) &nbsp;|&nbsp; {int(r.Hospitals):,} hospitals &nbsp;|&nbsp; Claim ratio {int(r['Claim ratio %'])}%
        <br>{tags}</div>""", unsafe_allow_html=True)
    st.download_button("⬇️ Download recommendations (CSV)",
                       top[["Plan", "Match %", "Cover", "Family premium", "Hospitals", "Claim ratio %"]].to_csv(index=False),
                       "recommended_plans.csv", "text/csv")

with t2:
    sel = st.selectbox("Compare your needs with", res.Plan.tolist())
    row = PLANS[PLANS.Plan == sel].iloc[0]
    us, ps = scale(uvec), scale(row[FEATS].values.astype(float))
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=list(us) + [us[0]], theta=FEATS + [FEATS[0]], fill="toself", name="Your needs"))
    fig.add_trace(go.Scatterpolar(r=list(ps) + [ps[0]], theta=FEATS + [FEATS[0]], fill="toself", name=sel))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 5])), height=450, transition={"duration": 700})
    c1, c2 = st.columns(2)
    c1.plotly_chart(fig, use_container_width=True)
    bar = px.bar(res, x="Match %", y="Plan", orientation="h", color="Within budget", title="All plans ranked",
                 color_discrete_map={True: "#22c55e", False: "#9ca3af"}, animation_frame=None)
    bar.update_layout(height=450, yaxis=dict(autorange="reversed"))
    c2.plotly_chart(bar, use_container_width=True)
    st.dataframe(res[["Plan", "Match %", "Cover", "Family premium", "Hospitals", "Claim ratio %", "Within budget"]],
                 use_container_width=True, hide_index=True)

with t3:
    best = PLANS[PLANS.Plan == top.iloc[0].Plan].iloc[0]
    contrib = contributions(uvec, best).sort_values()
    st.subheader(f"What drives the match with {best.Plan}")
    st.plotly_chart(px.bar(contrib, orientation="h", labels={"value": "Contribution to match (points)", "index": ""})
                    .update_layout(height=380, showlegend=False), use_container_width=True)
    st.caption("Each bar shows how much that feature of the plan lines up with your needs. Larger = stronger reason for the match.")
    heat = pd.DataFrame(cosine_similarity(scale(MAT)), index=PLANS.Plan, columns=PLANS.Plan)
    st.plotly_chart(px.imshow(heat, color_continuous_scale="Blues", title="How similar the plans are to each other")
                    .update_layout(height=520), use_container_width=True)

with t4:
    st.markdown("""
**Method:** your family profile and every plan are turned into numbers (cover, maternity, OPD, critical illness,
senior care, child care, affordability). The app measures the **cosine similarity** between your needs and each plan;
the closer the direction of the two vectors, the higher the match %. This is the same technique used in the
Cinemagic AI movie recommender.

**Premium:** family premium = base premium × (1 + 0.8 per extra adult + 0.35 per child + 1.1 per senior).

**Note:** plan data is illustrative (not from a real insurer). Replace `PLANS` with real plan data to use it in practice.
""")
