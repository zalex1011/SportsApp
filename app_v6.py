# app_v6.py  –  Full Demo Mode (χωρίς API κλήσεις)

import streamlit as st
import pandas as pd
import numpy as np
import base64
import pickle
import io
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# =============== PAGE CONFIG & DARK NEON STYLE ===============
st.set_page_config(page_title="Sports Predictions — Demo Pro UI", layout="wide")

NEON_CSS = """
<style>
:root{
  --bg:#0b1020;
  --panel:#11162b;
  --muted:#93a0c6;
  --txt:#e6e9ff;
  --neon:#00ffd1;
  --neon2:#ff4dff;
  --ok:#22c55e;
  --warn:#f59e0b;
  --bad:#ef4444;
}
html, body, [data-testid="stAppViewContainer"]{
  background:var(--bg)!important;color:var(--txt)!important;
}
[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.2rem; padding-bottom:2rem;}
h1,h2,h3{color:var(--txt);}
small, .muted{color:var(--muted);}

.neon-title{
  font-weight:800; font-size:1.6rem; letter-spacing:.5px;
  text-shadow:0 0 6px var(--neon), 0 0 12px rgba(0,255,209,.25);
}
.subtle{color:var(--muted);font-size:.9rem}

.card{
  background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border:1px solid rgba(0,255,209,.25);
  border-radius:16px; padding:16px; margin-bottom:14px;
  box-shadow:0 0 10px rgba(0,255,209,.08), inset 0 0 10px rgba(0,0,0,.25);
  animation:fadeIn .35s ease both;
}
@keyframes fadeIn {from{opacity:0; transform:translateY(6px)} to{opacity:1; transform:translateY(0)}}

.pill{
  display:inline-block; padding:5px 10px; border-radius:999px;
  background:rgba(255,255,255,.06); color:var(--txt); font-weight:700; font-size:.8rem; margin-right:8px;
  border:1px solid rgba(255,255,255,.1);
}
.kv{display:flex; gap:12px; flex-wrap:wrap;}
.kv .item{
  background:rgba(255,255,255,.04); padding:8px 10px;
  border-radius:10px; border:1px solid rgba(255,255,255,.06)
}

.label{color:var(--muted); font-weight:600; font-size:.85rem}
.pct{font-weight:800; letter-spacing:.2px}

.barwrap{
  width:100%; height:10px; border-radius:999px; overflow:hidden;
  background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.1)
}
.bar{
  height:100%; background:linear-gradient(90deg, var(--neon), var(--neon2));
  box-shadow:0 0 8px var(--neon), 0 0 8px var(--neon2);
}

.metric-box{
  background:var(--panel); border:1px solid rgba(255,255,255,.08);
  padding:14px; border-radius:12px; text-align:center;
  box-shadow:0 0 12px rgba(0,0,0,.25), inset 0 0 10px rgba(255,255,255,.03);
}
.metric-title{color:var(--muted); font-size:.9rem}
.metric-value{font-size:1.4rem; font-weight:800}

hr{border-color: rgba(255,255,255,.08)}
</style>
"""
st.markdown(NEON_CSS, unsafe_allow_html=True)

# =============== CONSTANTS ===============

# Λίγκες που θες (μόνο για το UI dropdown, DEMO δεν καλεί API)
LEAGUES = [
    ("Premier League (England)", 39),
    ("Super League Greece", 197),
    ("La Liga (Spain)", 140),
    ("Serie A (Italy)", 135),
    ("Challenger Pro League (Belgium - B Division)", 145),
    ("Pro League (Belgium - A Division)", 144),
    ("Super Lig (Turkey)", 203),
    ("Eredivisie (Netherlands)", 88),
    ("Primeira Liga (Portugal)", 94),
    ("Ligue 1 (France)", 61),
    ("Bundesliga (Germany)", 78),
    ("Super League 2 (Greece - B Division)", 494),
]
SEASONS = ["2025", "2024", "2023", "2022"]

DEMO_MATCHES_PER_RUN = 12  # όπως ζήτησες (option 2B)

# =============== LOAD MODELS (IF AVAILABLE) ===============
@st.cache_resource
def load_models_if_exist():
    """
    Προσπαθεί να φορτώσει τα πραγματικά μοντέλα.
    Αν δεν τα βρει, γυρίζει (None, None) και πάμε σε πλήρως demo mode.
    """
    model_result = None
    model_over = None
    try:
        with open("models/model_result_real.txt", "r") as f:
            result_bytes = base64.b64decode(f.read())
        model_result = pickle.load(io.BytesIO(result_bytes))
    except Exception:
        model_result = None

    try:
        with open("models/model_over_real.txt", "r") as f:
            over_bytes = base64.b64decode(f.read())
        model_over = pickle.load(io.BytesIO(over_bytes))
    except Exception:
        model_over = None

    return model_result, model_over

model_result, model_over = load_models_if_exist()

if model_result is None or model_over is None:
    st.warning("💡 Demo Mode: Δεν βρέθηκαν τα μοντέλα στο `models/`. "
               "Χρησιμοποιούνται demo πιθανότητες (τυχαίες αλλά ρεαλιστικές).")

# =============== HELPERS ===============

def color_for_prob(pct: float) -> str:
    if pct >= 70:
        return "#22c55e"
    if pct >= 55:
        return "#f59e0b"
    return "#ef4444"

def suggestion_from_result(probs: np.ndarray) -> str:
    labels = ["Home", "Draw", "Away"]
    return labels[int(np.argmax(probs))]

def suggestion_from_ou(probs: np.ndarray) -> str:
    return "Over 2.5" if probs[1] >= probs[0] else "Under 2.5"

def conf_result(probs: np.ndarray) -> float:
    top = float(np.max(probs))
    second = float(np.sort(probs)[-2])
    gap = max(0.0, top - second)
    conf = min(1.0, 0.75*gap + 0.25*(top - 1/len(probs)))
    return float(np.clip(conf, 0.0, 1.0))

def conf_ou(probs: np.ndarray) -> float:
    p_over = float(probs[1])
    return float(np.clip(2.0*abs(p_over - 0.5), 0.0, 1.0))

def generate_demo_fixtures(league_name: str, season: str, n_matches: int = DEMO_MATCHES_PER_RUN):
    """
    Δημιουργεί demo fixtures (Premier League style) για testing.
    Δεν χρησιμοποιεί API. Τα ονόματα είναι ρεαλιστικά.
    """
    # Premier League style teams
    pl_teams = [
        "Arsenal", "Chelsea", "Liverpool", "Manchester City",
        "Manchester United", "Tottenham", "Newcastle", "Aston Villa",
        "West Ham", "Brighton", "Brentford", "Wolves",
        "Everton", "Crystal Palace", "Leicester", "Leeds",
        "Nottingham Forest", "Fulham", "Bournemouth", "Southampton"
    ]
    rng = np.random.default_rng(abs(hash((league_name, season))) % (2**32))

    fixtures = []
    today = datetime.utcnow().date()
    for i in range(n_matches):
        home, away = rng.choice(pl_teams, size=2, replace=False)
        dt = today + timedelta(days=int(i/2))  # λίγη διάσπαση στις ημερομηνίες
        fixtures.append({
            "date": dt.strftime("%Y-%m-%d"),
            "home": home,
            "away": away
        })
    return fixtures

def get_probs_for_match(home: str, away: str):
    """
    Αν υπάρχουν πραγματικά μοντέλα, τα χρησιμοποιεί σε dummy features.
    Αλλιώς, γυρίζει πλήρως demo πιθανότητες.
    """
    seed_val = abs(hash((home, away))) % (2**32)
    rng = np.random.default_rng(seed_val)

    # PREDICT RESULT
    if model_result is not None:
        n_res = getattr(model_result, "n_features_in_", 20)
        X_res = rng.random((1, n_res))
        probs_res = model_result.predict_proba(X_res)[0]  # [Home, Draw, Away]
    else:
        raw = rng.random(3)
        probs_res = raw / raw.sum()

    # PREDICT OVER/UNDER
    if model_over is not None:
        n_ou = getattr(model_over, "n_features_in_", 20)
        X_ou = rng.random((1, n_ou))
        probs_ou = model_over.predict_proba(X_ou)[0]      # [Under, Over]
    else:
        p_over = rng.uniform(0.3, 0.7)
        probs_ou = np.array([1-p_over, p_over])

    return probs_res, probs_ou

# =============== TABS ===============
tab_pred, tab_train = st.tabs(["🎴 Προβλέψεις (Demo)", "📊 Training Dashboard (Demo)"])

# ====================== TAB: PREDICTIONS (DEMO) ======================
with tab_pred:
    st.markdown('<div class="neon-title">Sports Predictions — Demo Card View</div>', unsafe_allow_html=True)
    st.caption("Demo mode χωρίς API. Ρεαλιστικά ονόματα ομάδων, κάρτες προβλέψεων, φερεγγυότητα & CSV export.")

    colA, colB, colC = st.columns([2,1,1])
    with colA:
        league_label = st.selectbox("Λίγκα", [f"{name} ({lid})" for name, lid in LEAGUES])
        league_name = league_label.split("(")[0].strip()
    with colB:
        season = st.selectbox("Σεζόν (Demo)", SEASONS, index=0)
    with colC:
        days_ahead = st.slider("Μέρες μπροστά (Demo)", 1, 14, 7,
                               help="Στο Demo δεν αλλάζει τα fixtures, απλώς υπάρχει για ρεαλισμό.")

    go = st.button("🔮 Δημιούργησε προβλέψεις (Demo)", use_container_width=True)

    if go:
        fixtures = generate_demo_fixtures(league_name, season, DEMO_MATCHES_PER_RUN)

        cards_data = []
        for f in fixtures:
            home = f["home"]
            away = f["away"]
            date = f["date"]

            probs_res, probs_ou = get_probs_for_match(home, away)

            pick_res = suggestion_from_result(probs_res)
            pick_ou  = suggestion_from_ou(probs_ou)

            c_res = conf_result(probs_res)
            c_ou  = conf_ou(probs_ou)
            c_comb = float(np.clip((c_res + c_ou)/2, 0.0, 1.0))

            row = {
                "Ημερομηνία": date,
                "Γηπεδούχος": home,
                "Φιλοξενούμενος": away,
                "P_Home": round(probs_res[0]*100,1),
                "P_Draw": round(probs_res[1]*100,1),
                "P_Away": round(probs_res[2]*100,1),
                "P_Over": round(probs_ou[1]*100,1),
                "P_Under": round(probs_ou[0]*100,1),
                "Pick_Result": pick_res,
                "Pick_OU": pick_ou,
                "Conf_Result": round(c_res*100,1),
                "Conf_OU": round(c_ou*100,1),
                "Conf_Combined": round(c_comb*100,1),
            }
            cards_data.append(row)

        overall = round(np.mean([c["Conf_Combined"] for c in cards_data]), 1)
        st.markdown(f"**Συνολική Φερεγγυότητα (Result + Over/Under): {overall}% (Demo)**")
        st.markdown(f"""
            <div class="barwrap"><div class="bar" style="width:{overall}%"></div></div>
        """, unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)

        # Render cards 2 columns
        cols = st.columns(2)
        for i, c in enumerate(cards_data):
            with cols[i % 2]:
                color_home = color_for_prob(c["P_Home"])
                color_draw = color_for_prob(c["P_Draw"])
                color_away = color_for_prob(c["P_Away"])
                color_ov   = color_for_prob(c["P_Over"])
                color_un   = color_for_prob(c["P_Under"])

                st.markdown(f"""
                <div class="card">
                  <div class="subtle">{c["Ημερομηνία"]} • {league_name} • Season {season}</div>
                  <div class="neon-title" style="font-size:1.15rem">{c["Γηπεδούχος"]} <span class="subtle">vs</span> {c["Φιλοξενούμενος"]}</div>
                  <div style="margin-top:6px; margin-bottom:10px;">
                    <span class="pill">{c["Pick_Result"]} • Result</span>
                    <span class="pill">{c["Pick_OU"]} • O/U</span>
                  </div>
                  <div class="kv">
                    <div class="item"><span class="label">Home:</span> <span class="pct" style="color:{color_home}">{c["P_Home"]}%</span></div>
                    <div class="item"><span class="label">Draw:</span> <span class="pct" style="color:{color_draw}">{c["P_Draw"]}%</span></div>
                    <div class="item"><span class="label">Away:</span> <span class="pct" style="color:{color_away}">{c["P_Away"]}%</span></div>
                  </div>
                  <div class="kv" style="margin-top:6px;">
                    <div class="item"><span class="label">Over 2.5:</span> <span class="pct" style="color:{color_ov}">{c["P_Over"]}%</span></div>
                    <div class="item"><span class="label">Under 2.5:</span> <span class="pct" style="color:{color_un}">{c["P_Under"]}%</span></div>
                  </div>
                  <div class="subtle" style="margin-top:10px">Φερεγγυότητα Αποτελέσματος: {c["Conf_Result"]}% • Φερεγγυότητα O/U: {c["Conf_OU"]}%</div>
                  <div style="margin-top:6px;">Συνδυαστική Φερεγγυότητα</div>
                  <div class="barwrap" title="{c["Conf_Combined"]}%">
                    <div class="bar" style="width:{c["Conf_Combined"]}%"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # Download CSV
        out_df = pd.DataFrame(cards_data)
        csv = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Κατέβασε CSV (Demo)", csv, "predictions_cards_demo.csv",
                           "text/csv", use_container_width=True)

# ====================== TAB: TRAINING DASHBOARD (DEMO) ======================
with tab_train:
    st.markdown('<div class="neon-title">Training & Model Evaluation Dashboard (Demo)</div>', unsafe_allow_html=True)
    st.caption("Δοκιμαστικό ταμπλό με συνθετικά metrics. Όταν ενεργοποιήσεις paid API + ιστορικά δεδομένα, "
               "θα συνδέσουμε εδώ πραγματικά training αποτελέσματα.")

    left, right = st.columns([1,1])
    with left:
        seed = st.number_input("Seed (Demo)", min_value=0, value=42, step=1,
                               help="Αλλάζεις το seed για νέο demo run.")
        epochs = st.slider("Epochs (Demo)", 5, 50, 15)
    with right:
        classes = ["Home","Draw","Away"]
        st.write("Κλάσεις Αποτελέσματος:", ", ".join(classes))
        st.info("Στο πραγματικό training θα δείχνουμε metrics από το script εκπαίδευσης σου (π.χ. predictor_real_v3.py).")

    rng = np.random.default_rng(seed)
    # Demo metrics
    train_acc = np.clip(np.cumsum(rng.normal(0.02, 0.01, epochs)) + 0.6, 0.6, 0.98)
    val_acc   = np.clip(train_acc - rng.normal(0.03, 0.015, epochs), 0.5, 0.96)
    train_loss = np.clip(np.linspace(1.2, 0.4, epochs) + rng.normal(0, 0.05, epochs), 0.3, 1.5)
    val_loss   = np.clip(train_loss + rng.normal(0.05, 0.06, epochs), 0.35, 1.6)

    acc_now = round(float(val_acc[-1])*100, 1)
    loss_now = round(float(val_loss[-1]), 3)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f'<div class="metric-box"><div class="metric-title">Validation Accuracy (Demo)</div>'
            f'<div class="metric-value">{acc_now}%</div></div>', unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f'<div class="metric-box"><div class="metric-title">Validation Loss (Demo)</div>'
            f'<div class="metric-value">{loss_now}</div></div>', unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            f'<div class="metric-box"><div class="metric-title">Epochs</div>'
            f'<div class="metric-value">{epochs}</div></div>', unsafe_allow_html=True
        )

    st.markdown("<br/>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📈 Accuracy Curve (Demo)")
        fig, ax = plt.subplots()
        ax.plot(range(1,epochs+1), train_acc, label="Train", linewidth=2)
        ax.plot(range(1,epochs+1), val_acc, label="Validation", linewidth=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.legend()
        ax.grid(alpha=.2)
        st.pyplot(fig)

    with c2:
        st.subheader("📉 Loss Curve (Demo)")
        fig2, ax2 = plt.subplots()
        ax2.plot(range(1,epochs+1), train_loss, label="Train", linewidth=2)
        ax2.plot(range(1,epochs+1), val_loss, label="Validation", linewidth=2)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.legend()
        ax2.grid(alpha=.2)
        st.pyplot(fig2)

    st.subheader("🧩 Confusion Matrix (Demo)")
    cm = rng.integers(20, 180, size=(3,3))
    fig3, ax3 = plt.subplots()
    im = ax3.imshow(cm, cmap="viridis")
    ax3.set_xticks(range(3)); ax3.set_yticks(range(3))
    ax3.set_xticklabels(classes); ax3.set_yticklabels(classes)
    for i in range(3):
        for j in range(3):
            ax3.text(j, i, cm[i, j], ha="center", va="center", color="white", fontsize=10)
    ax3.set_xlabel("Predicted"); ax3.set_ylabel("Actual")
    fig3.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    st.pyplot(fig3)

    st.info("Όταν είσαι έτοιμος με paid API + ιστορικά δεδομένα, "
            "θα αντικαταστήσουμε τα demo metrics με πραγματικά αποτελέσματα εκπαίδευσης.")
