# app_v5.py
import streamlit as st
import pandas as pd
import numpy as np
import base64, pickle, requests, io, os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# =============== PAGE CONFIG & DARK NEON STYLE ===============
st.set_page_config(page_title="Sports Predictions — Pro UI", layout="wide")

NEON_CSS = """
<style>
:root{
  --bg:#0b1020;          /* deep dark */
  --panel:#11162b;       /* card bg */
  --muted:#93a0c6;       /* muted text */
  --txt:#e6e9ff;         /* main text */
  --neon:#00ffd1;        /* cyan neon */
  --neon2:#ff4dff;       /* magenta neon */
  --ok:#22c55e;          /* green */
  --warn:#f59e0b;        /* amber */
  --bad:#ef4444;         /* red */
}
html, body, [data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--txt)!important;}
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
.kv .item{background:rgba(255,255,255,.04); padding:8px 10px; border-radius:10px; border:1px solid rgba(255,255,255,.06)}

.label{color:var(--muted); font-weight:600; font-size:.85rem}
.pct{font-weight:800; letter-spacing:.2px}

.barwrap{width:100%; height:10px; border-radius:999px; overflow:hidden; background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.1)}
.bar{height:100%; background:linear-gradient(90deg, var(--neon), var(--neon2)); box-shadow:0 0 8px var(--neon), 0 0 8px var(--neon2);}

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
API_KEY = st.secrets.get("API_KEY") or os.environ.get("API_KEY", "")
if not API_KEY:
    st.error("🔑 Βάλε το API key στα Secrets (Manage app → Settings → Secrets) με κλειδί: `API_KEY`")
    st.stop()

BASE_URL = "https://v3.football.api-sports.io/fixtures"
HEADERS = {"x-apisports-key": API_KEY}

# Λίγκες όπως ζητήθηκαν (χειροκίνητη λίστα)
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
    ("Super League 2 (Greece - B Division)", 494)
]
SEASONS = ["2025", "2024", "2023", "2022"]
@st.cache_data
def get_current_season(league_id: int):
    url = "https://v3.football.api-sports.io/leagues"
    params = {"league": league_id}
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    data = r.json()
    seasons = data["response"][0]["seasons"]

    for s in seasons:
        if s["current"] == True:
            return s["year"]  # example: 2025

    return seasons[-1]["year"]  # fallback

# =============== LOAD MODELS FROM BASE64 TXT ===============
@st.cache_resource
def load_models():
    try:
        with open("models/model_result_real.txt", "r") as f:
            result_bytes = base64.b64decode(f.read())
        with open("models/model_over_real.txt", "r") as f:
            over_bytes = base64.b64decode(f.read())
    except FileNotFoundError as e:
        st.error("❌ Λείπουν τα base64 αρχεία μοντέλων στο `models/` (model_result_real.txt, model_over_real.txt).")
        raise e
    model_result = pickle.load(io.BytesIO(result_bytes))  # multi-class: [Home, Draw, Away]
    model_over = pickle.load(io.BytesIO(over_bytes))      # binary: [Under, Over]
    return model_result, model_over

model_result, model_over = load_models()

# =============== HELPERS ===============
def fetch_fixtures(league_id: int, season: int, days_ahead: int):
    current_season = get_current_season(league_id)

    # === CASE 1: ΤΡΕΧΟΥΣΑ ΣΕΖΟΝ ===
    if season == current_season:
        today = datetime.utcnow().date()
        future = today + timedelta(days=days_ahead)

        params = {
            "league": league_id,
            "season": season,
            "from": str(today),
            "to": str(future)
        }
        r = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)
        return r.json().get("response", [])

    # === CASE 2: ΠΑΛΙΑ ΣΕΖΟΝ ===
    elif season < current_season:
        start_date = f"{season}-08-01"
        end_date = f"{season+1}-06-30"

        params = {
            "league": league_id,
            "season": season,
            "from": start_date,
            "to": end_date
        }
        r = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)
        return r.json().get("response", [])

    # === CASE 3: ΜΕΛΛΟΝΤΙΚΗ ΣΕΖΟΝ ===
    else:
        st.warning("⚠️ Η σεζόν αυτή δεν έχει ξεκινήσει ακόμη.")
        return []

def color_for_prob(pct: float) -> str:
    if pct >= 70: return "#22c55e"  # green
    if pct >= 55: return "#f59e0b"  # amber
    return "#ef4444"                # red

def suggestion_from_result(probs: np.ndarray) -> str:
    labels = ["Home", "Draw", "Away"]
    return labels[int(np.argmax(probs))]

def suggestion_from_ou(probs: np.ndarray) -> str:
    return "Over 2.5" if probs[1] >= probs[0] else "Under 2.5"

def conf_result(probs: np.ndarray) -> float:
    top = float(np.max(probs)); second = float(np.sort(probs)[-2])
    gap = max(0.0, top - second)  # 0..1
    conf = min(1.0, 0.75*gap + 0.25*(top - 1/len(probs)))
    return float(np.clip(conf, 0.0, 1.0))

def conf_ou(probs: np.ndarray) -> float:
    p_over = float(probs[1])
    return float(np.clip(2.0*abs(p_over - 0.5), 0.0, 1.0))

# =============== TABS ===============
tab_pred, tab_train = st.tabs(["🎴 Προβλέψεις", "📊 Training Dashboard"])

# ====================== TAB: PREDICTIONS ======================
with tab_pred:
    st.markdown('<div class="neon-title">Sports Predictions — Card View (AI Mode)</div>', unsafe_allow_html=True)
    st.caption("Dark Neon UI • Προτάσεις & Φερεγγυότητα (Result & Over/Under) • Download CSV")

    colA, colB, colC = st.columns([2,1,1])
    with colA:
        league_label = st.selectbox("Λίγκα", [f"{name} ({lid})" for name, lid in LEAGUES])
        league_id = int(league_label.split("(")[-1].rstrip(")"))
    with colB:
        season = st.selectbox("Σεζόν", SEASONS, index=0)
    with colC:
        days_ahead = st.slider("Μέρες μπροστά", 1, 14, 7)

    go = st.button("🔮 Δημιούργησε προβλέψεις", use_container_width=True)

    if go:
        with st.spinner("📡 Ανάκτηση αγώνων από API..."):
            fixtures = fetch_fixtures(league_id, season, days_ahead)

        if not fixtures:
            st.warning("⚠️ Δεν βρέθηκαν επερχόμενοι αγώνες για τις επιλεγμένες ρυθμίσεις.")
            st.stop()

        cards_data = []
        for m in fixtures:
            home = m["teams"]["home"]["name"]
            away = m["teams"]["away"]["name"]
            date = m["fixture"]["date"]
            short_date = date[:16].replace("T", " ")

            # Dummy features για σταθερή demo λειτουργία (θα τα αντικαταστήσουμε με real features όταν ενεργοποιηθεί το paid API)
            n_res = getattr(model_result, "n_features_in_", 20)
            n_ou  = getattr(model_over, "n_features_in_", 20)

            seed_val = abs(hash((home, away))) % (2**32)
            rng_res = np.random.default_rng(seed_val)
            rng_ou  = np.random.default_rng(seed_val + 13)
            X_res = rng_res.random((1, n_res))
            X_ou  = rng_ou.random((1, n_ou))

            probs_res = model_result.predict_proba(X_res)[0]  # [Home, Draw, Away]
            probs_ou  = model_over.predict_proba(X_ou)[0]     # [Under, Over]

            pick_res = suggestion_from_result(probs_res)
            pick_ou  = suggestion_from_ou(probs_ou)

            c_res = conf_result(probs_res)
            c_ou  = conf_ou(probs_ou)
            c_comb = float(np.clip((c_res + c_ou)/2, 0.0, 1.0))

            row = {
                "Ημερομηνία": short_date,
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

        # Overall reliability
        overall = round(np.mean([c["Conf_Combined"] for c in cards_data]), 1)
        st.markdown(f"**Συνολική Φερεγγυότητα (Result + Over/Under): {overall}%**")
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
                  <div class="subtle">{c["Ημερομηνία"]}</div>
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
        st.download_button("📥 Κατέβασε CSV", csv, "predictions_cards.csv", "text/csv", use_container_width=True)

# ====================== TAB: TRAINING DASHBOARD (DEMO) ======================
with tab_train:
    st.markdown('<div class="neon-title">Training & Model Evaluation Dashboard (Demo)</div>', unsafe_allow_html=True)
    st.caption("Δοκιμαστικό ταμπλό με συνθετικά metrics. Όταν ενεργοποιήσεις paid API + ιστορικά δεδομένα, μπορούμε να συνδέσουμε πραγματικά training αποτελέσματα.")

    left, right = st.columns([1,1])
    with left:
        seed = st.number_input("Seed", min_value=0, value=42, step=1, help="Αλλάζεις το seed για νέο demo run.")
        epochs = st.slider("Epochs", 5, 50, 15)
    with right:
        classes = ["Home","Draw","Away"]
        st.write("Κλάσεις Αποτελέσματος:", ", ".join(classes))
        st.info("Στο πραγματικό training θα δείχνουμε metrics από το `predictor_real_v3.py`.")

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
        st.markdown('<div class="metric-box"><div class="metric-title">Validation Accuracy</div>'
                    f'<div class="metric-value">{acc_now}%</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-box"><div class="metric-title">Validation Loss</div>'
                    f'<div class="metric-value">{loss_now}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-box"><div class="metric-title">Epochs</div>'
                    f'<div class="metric-value">{epochs}</div></div>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📈 Accuracy Curve")
        fig, ax = plt.subplots()
        ax.plot(range(1,epochs+1), train_acc, label="Train", linewidth=2)
        ax.plot(range(1,epochs+1), val_acc, label="Validation", linewidth=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.legend()
        ax.grid(alpha=.2)
        st.pyplot(fig)

    with c2:
        st.subheader("📉 Loss Curve")
        fig2, ax2 = plt.subplots()
        ax2.plot(range(1,epochs+1), train_loss, label="Train", linewidth=2)
        ax2.plot(range(1,epochs+1), val_loss, label="Validation", linewidth=2)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.legend()
        ax2.grid(alpha=.2)
        st.pyplot(fig2)

    st.subheader("🧩 Confusion Matrix (Demo)")
    # 3x3 CM
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

    st.info("Όταν ενεργοποιήσεις paid API και ανεβάσουμε τα ιστορικά CSV/features, το tab αυτό θα δείχνει πραγματικά metrics από το training script σου.")
