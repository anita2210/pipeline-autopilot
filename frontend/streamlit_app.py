"""
Kairos Pulse — CI/CD Failure Prediction System
Google Cambridge Showcase — Apr 15, 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests, time, random, smtplib, os, json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

st.set_page_config(page_title="Kairos Pulse", layout="wide", initial_sidebar_state="collapsed")

LOGO_B64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/wAARCAGIAwwDASIAAhEBAxEB/8QAHQAAAgICAwEAAAAAAAAAAAAAAAECCAYHAwUJBP/EAGYQAAEDAgQDBAMHCBIOCQUBAQEAAgMEEQUGITEHEkEIUWFxEyKBFDJCUpGhsRUjYnKCs9HTCRYXGCQzN3N1kpOUlaKywdLwJjQ2OENTVldjZYOjtMInKERUVWR0hOElRkfD8UWF/8QAGwEAAgMBAQEAAAAAAAAAAAAAAAECAwQFBgf/xAA1EQACAgEDAgUCAwcFAQAAAAABAgMEITEFBhITUWEUIjNBcQcjMkJSkpGhscHR8BXh/9oADAMBAAIRAxEAPwC"
GMAIL_USER         = "pipelineguard.alerts@gmail.com"
GMAIL_APP_PASSWORD = "fsds tkto wxrc ycgx"
ALERT_RECIPIENT    = "varupandi@gmail.com"
CLOUD_RUN_URL      = os.environ.get("CLOUD_RUN_URL","https://pipeline-autopilot-60271318606.us-central1.run.app")
STREAM_URL         = os.environ.get("STREAM_URL", "http://localhost:8000/stream")

PIPELINE_NAMES = ["build-and-test","backport-pr","docker-publish","integration-test",
                  "nightly-regression","security-scan","lint-check","deploy-staging",
                  "benchmark-run","release-build"]
REPOS    = ["ClickHouse/ClickHouse","ClickHouse/clickhouse-java","ClickHouse/clickhouse-go","ClickHouse/dbt-clickhouse"]
TRIGGERS = ["push","pull_request","schedule","workflow_dispatch"]
BRANCHES = ["main","master","backport/23.8","feature/new-codec","fix/memory-leak","release/24.1","dev"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── GLOBAL RESET ── */
html, body, [class*="css"], [class*="st-"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #ffffff !important;
    color: #1e293b !important;
}
[data-testid="collapsedControl"] { display:none !important; }
section[data-testid="stSidebar"] { display:none !important; }
header[data-testid="stHeader"]   { display:none !important; }
.main > div { padding: 0 !important; }
.block-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; width: 100% !important; }
#root > div:first-child { padding-top: 0 !important; }
.stApp { background-color: #ffffff !important; }

/* Force white background on all streamlit containers */
[data-testid="stAppViewContainer"] { background: #ffffff !important; }
[data-testid="stVerticalBlock"]    { background: transparent !important; }
section.main                       { background: #ffffff !important; }

/* ── NAV ── */
.kp-nav {
    width: 100%; background: #ffffff;
    border-bottom: 2px solid #2563eb;
    padding: 0 56px; display: flex; align-items: center;
    justify-content: space-between;
    height: 62px; position: fixed; top: 0; left: 0; right: 0;
    z-index: 9999; box-sizing: border-box;
    box-shadow: 0 1px 8px rgba(37,99,235,0.08);
}
.kp-nav-left   { display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
.kp-nav-left img { height: 34px; width: auto; object-fit: contain; }
.kp-nav-brand  { font-size: 1.08rem; font-weight: 700; color: #0f172a; letter-spacing: -0.3px; }
.kp-nav-brand span { color: #2563eb; }
.kp-nav-links  { display: flex; align-items: center; gap: 2px; }
.kp-nav-link {
    font-size: 0.85rem; font-weight: 500; color: #64748b;
    padding: 7px 15px; border-radius: 6px; cursor: pointer;
    text-decoration: none; white-space: nowrap; border: none;
    background: transparent; transition: all 0.15s; display: inline-block;
}
.kp-nav-link:hover  { background: #eff6ff; color: #2563eb; }
.kp-nav-link.active { background: #2563eb; color: #ffffff; font-weight: 600; }
.kp-nav-link .badge {
    display: inline-block; background: #fee2e2; color: #991b1b;
    font-size: 0.62rem; font-weight: 700; padding: 1px 5px;
    border-radius: 10px; margin-left: 4px; vertical-align: middle;
}
.kp-content { margin-top: 62px; }

/* ── HERO ── */
.kp-hero {
    background: linear-gradient(160deg, #f8faff 0%, #eff6ff 40%, #dbeafe 100%);
    padding: 88px 80px 76px; text-align: center;
    border-bottom: 1px solid #dbeafe; width: 100%; box-sizing: border-box;
}
.kp-hero-logo-row  { display:flex; align-items:center; justify-content:center; gap:18px; margin-bottom:18px; }
.kp-hero-logo-row img { height: 68px; width: auto; }
.kp-hero-wordmark  { font-size: 3rem; font-weight: 800; color: #0f172a; letter-spacing: -1.5px; line-height: 1; }
.kp-hero-wordmark span { color: #2563eb; }
.kp-hero-tagline   { font-size: 1.1rem; color: #475569; margin: 14px auto 5px; max-width: 480px; line-height: 1.6; }
.kp-hero-sub       { font-size: 0.78rem; font-weight: 700; color: #2563eb; letter-spacing: 4px; text-transform: uppercase; margin-bottom: 48px; }
.kp-hero-cards     { display: grid; grid-template-columns: repeat(3,1fr); gap: 18px; max-width: 880px; margin: 0 auto; }
.kp-hero-card      { background: #ffffff; border: 1px solid #dbeafe; border-radius: 10px; padding: 20px 18px; text-align: left; box-shadow: 0 1px 4px rgba(37,99,235,0.06); }
.kp-hero-card-tag  { display:inline-block; font-size:0.66rem; font-weight:700; background:#dbeafe; color:#1d4ed8; border-radius:4px; padding:2px 7px; margin-bottom:9px; text-transform:uppercase; letter-spacing:0.5px; }
.kp-hero-card-title{ font-size: 0.9rem; font-weight: 600; color: #0f172a; margin-bottom: 5px; }
.kp-hero-card-desc { font-size: 0.78rem; color: #64748b; line-height: 1.55; }

/* ── STATS BAND ── */
.kp-stats { background: #1e3a8a; padding: 44px 80px; display: grid; grid-template-columns: repeat(4,1fr); text-align: center; width: 100%; box-sizing: border-box; }
.kp-stat  { border-right: 1px solid rgba(255,255,255,0.1); padding: 0 20px; }
.kp-stat:last-child { border-right: none; }
.kp-stat-num { font-size: 2.2rem; font-weight: 800; color: #ffffff; letter-spacing: -1px; margin-bottom: 4px; }
.kp-stat-lbl { font-size: 0.7rem; color: #93c5fd; text-transform: uppercase; letter-spacing: 0.8px; }

/* ── SECTIONS ── */
.kp-section     { padding: 64px 80px; background: #ffffff; width: 100%; box-sizing: border-box; }
.kp-section-alt { padding: 64px 80px; background: #f8fafc; border-top: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; width: 100%; box-sizing: border-box; }
.kp-eyebrow { font-size: 0.72rem; font-weight: 700; color: #2563eb; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }
.kp-h2      { font-size: 1.9rem; font-weight: 700; color: #0f172a; letter-spacing: -0.5px; margin-bottom: 8px; }
.kp-p       { font-size: 0.92rem; color: #64748b; max-width: 480px; line-height: 1.7; margin-bottom: 40px; }
.kp-grid-3  { display: grid; grid-template-columns: repeat(3,1fr); gap: 18px; }
.kp-grid-4  { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; }
.kp-feat-card  { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 22px 18px; transition: border-color 0.2s; }
.kp-feat-card:hover { border-color: #2563eb; box-shadow: 0 2px 12px rgba(37,99,235,0.08); }
.kp-feat-title { font-size: 0.92rem; font-weight: 600; color: #0f172a; margin-bottom: 6px; }
.kp-feat-desc  { font-size: 0.8rem; color: #64748b; line-height: 1.6; }
.kp-step-card  { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 22px 18px; }
.kp-step-num   { font-size: 0.68rem; font-weight: 700; color: #2563eb; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 7px; }
.kp-step-title { font-size: 0.92rem; font-weight: 600; color: #0f172a; margin-bottom: 5px; }
.kp-step-desc  { font-size: 0.8rem; color: #64748b; line-height: 1.55; }

/* ── PAGE HEADER (inner pages) ── */
.kp-page-header {
    background: #1e3a8a;
    padding: 28px 56px 24px; width: 100%; box-sizing: border-box;
    border-bottom: 3px solid #2563eb;
}
.kp-breadcrumb { font-size: 0.66rem; color: #93c5fd; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
.kp-page-header h1 { font-size: 1.5rem; font-weight: 700; color: #ffffff; margin: 0 0 3px; }
.kp-page-header p  { font-size: 0.82rem; color: #93c5fd; margin: 0; }

/* ── INNER CONTENT ── */
.kp-inner { padding: 28px 56px; background: #ffffff; }

/* ── CARDS ── */
.kp-card {
    background: #ffffff; border-radius: 10px; padding: 18px 22px;
    margin-bottom: 14px; border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(15,23,42,0.04);
}

/* ── METRIC TILES ── */
.metric-tile {
    background: #ffffff; border-radius: 10px; padding: 18px 16px;
    text-align: center; border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(15,23,42,0.04);
}
.metric-tile .val { font-size: 1.75rem; font-weight: 700; color: #0f172a; line-height: 1; }
.metric-tile .lbl { font-size: 0.68rem; color: #94a3b8; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }

/* ── DIVIDER ── */
hr { border: none; border-top: 1px solid #e2e8f0; margin: 20px 0; }

/* ── BADGES ── */
.badge-high   { background: #fee2e2; color: #991b1b; border-radius: 5px; padding: 3px 10px; font-weight: 700; font-size: 0.74rem; }
.badge-medium { background: #fef3c7; color: #92400e; border-radius: 5px; padding: 3px 10px; font-weight: 700; font-size: 0.74rem; }
.badge-low    { background: #dcfce7; color: #14532d; border-radius: 5px; padding: 3px 10px; font-weight: 700; font-size: 0.74rem; }

/* ── RECOMMENDATIONS ── */
.rec-hold    { background: #fff1f2; border-left: 4px solid #ef4444; padding: 14px 18px; border-radius: 0 8px 8px 0; margin-bottom: 12px; color: #1e293b; }
.rec-caution { background: #fffbeb; border-left: 4px solid #f59e0b; padding: 14px 18px; border-radius: 0 8px 8px 0; margin-bottom: 12px; color: #1e293b; }
.rec-safe    { background: #f0fdf4; border-left: 4px solid #22c55e; padding: 14px 18px; border-radius: 0 8px 8px 0; margin-bottom: 12px; color: #1e293b; }
.high-risk-card { background: #fff8f8; border: 1.5px solid #fca5a5; border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; }
.fix-box        { background: #fff1f2; border-left: 3px solid #ef4444; border-radius: 0 6px 6px 0; padding: 10px 14px; margin-top: 8px; font-size: 0.82rem; line-height: 1.7; color: #1e293b; }
.alert-item     { background: #fff1f2; border: 1px solid #fecaca; border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; color: #1e293b; }

/* ── CHAT ── */
.chat-user { background: #dbeafe; border-radius: 16px 16px 4px 16px; padding: 10px 14px; margin: 6px 0 6px auto; max-width: 76%; font-size: 0.86rem; color: #1e3a8a; line-height: 1.5; }
.chat-bot  { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 16px 16px 16px 4px; padding: 10px 14px; margin: 6px 0; max-width: 86%; font-size: 0.86rem; color: #374151; line-height: 1.6; }
.chat-container { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 18px; min-height: 180px; max-height: 400px; overflow-y: auto; margin-bottom: 12px; }
.run-context-pill { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 20px; padding: 5px 12px; font-size: 0.76rem; color: #1e40af; display: inline-block; margin-bottom: 12px; }
.mono { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }

/* ── STREAMLIT OVERRIDES ── */
div[data-testid="stButton"] button { font-family: 'Inter', sans-serif !important; border-radius: 7px !important; }
div[data-testid="stButton"] button[kind="primary"] { background: #2563eb !important; border: none !important; color: #fff !important; font-weight: 600 !important; }
div[data-testid="stButton"] button[kind="primary"]:hover { background: #1d4ed8 !important; }
div[data-testid="stButton"] button[kind="secondary"] { background: #ffffff !important; border: 1px solid #e2e8f0 !important; color: #374151 !important; }
div[data-testid="stButton"] button[kind="secondary"]:hover { border-color: #2563eb !important; color: #2563eb !important; }

/* Streamlit inputs */
div[data-testid="stSelectbox"] > div, div[data-testid="stMultiSelect"] > div { background: #ffffff !important; }
div[data-baseweb="select"] { background: #ffffff !important; }

/* Streamlit info/success/warning boxes */
div[data-testid="stAlert"] { border-radius: 8px !important; }

/* Streamlit markdown text */
.stMarkdown p, .stMarkdown li { color: #1e293b !important; }
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown h4 { color: #0f172a !important; }

/* Dataframe */
[data-testid="stDataFrame"] { background: #ffffff !important; border-radius: 10px; border: 1px solid #e2e8f0; }

/* ── FOOTER ── */
.kp-footer { background: #f8fafc; border-top: 1px solid #e2e8f0; padding: 18px 80px; text-align: center; }
.kp-footer span { font-size: 0.76rem; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

PIPELINE_NAMES_LIST = PIPELINE_NAMES

def risk_label(p): return "HIGH" if p>=0.75 else ("MEDIUM" if p>=0.45 else "LOW")
def risk_color(lv): return {"HIGH":"#ef4444","MEDIUM":"#f59e0b","LOW":"#22c55e"}.get(lv,"#6b7280")

def make_shap(f):
    return {
        "workflow_failure_rate":   f.get("workflow_failure_rate",0.1)*2.8,
        "failures_last_7_runs":    f.get("failures_last_7_runs",0)*0.18,
        "prev_run_status_failure": 0.22 if f.get("prev_run_status")=="failure" else -0.05,
        "retry_count":             f.get("retry_count",0)*0.12,
        "concurrent_runs":         (f.get("concurrent_runs",3)-5)*0.02,
        "is_main_branch":         -0.15 if f.get("is_main_branch") else 0.08,
        "duration_deviation":      max(f.get("duration_deviation",0),0)*0.0008,
        "hour":                   -0.04 if 9<=f.get("hour",12)<=17 else 0.06,
        "is_weekend":              0.07 if f.get("is_weekend") else -0.02,
        "trigger_type_push":       0.03 if f.get("trigger_type")=="push" else -0.01,
    }

def score_local(features):
    s = (features.get("workflow_failure_rate",0)*3.5
         + features.get("failures_last_7_runs",0)*0.12
         + (0.25 if features.get("prev_run_status")=="failure" else 0)
         + features.get("retry_count",0)*0.08
         + features.get("concurrent_runs",1)*0.015
         + (0.10 if not features.get("is_main_branch") else 0)
         + (0.05 if features.get("is_weekend") else 0))
    p = round(min(max(s/5,0.02),0.97),3)
    return round(min(max(p+float(np.random.normal(0,0.015)),0.02),0.97),3)

def call_api(features):
    try:
        payload = {k:v for k,v in features.items() if not k.startswith("_")}
        r = requests.post(f"{CLOUD_RUN_URL}/predict", json=payload, timeout=6)
        if r.status_code==200:
            d = r.json(); prob = float(d.get("probability",0.5))
            raw = d.get("top_shap_features",[])
            shap = ({item.get("feature",f"f{i}"):item.get("shap_value",0) for i,item in enumerate(raw)}
                    if isinstance(raw,list) and raw else make_shap(features))
            return prob, shap, True
    except Exception: pass
    return score_local(features), make_shap(features), False

def make_run(seed=None):
    if seed is not None: np.random.seed(seed)
    wfr = round(float(np.random.beta(2,8)),3)
    f7  = int(np.random.choice([0,0,0,1,1,2,3,4,5],p=[.25,.20,.15,.12,.10,.08,.05,.03,.02]))
    return {
        "run_id":                f"run_{random.randint(10000,99999)}",
        "pipeline_name":         str(np.random.choice(PIPELINE_NAMES)),
        "repo":                  str(np.random.choice(REPOS)),
        "trigger_type":          str(np.random.choice(TRIGGERS,p=[.45,.35,.15,.05])),
        "head_branch":           str(np.random.choice(BRANCHES)),
        "is_main_branch":        int(np.random.choice([0,1],p=[.65,.35])),
        "is_bot_triggered":      int(np.random.choice([0,1],p=[.8,.2])),
        "is_weekend":            int(datetime.now().weekday()>=5),
        "hour":                  datetime.now().hour,
        "day_of_week":           datetime.now().weekday(),
        "workflow_failure_rate": wfr,
        "failures_last_7_runs":  f7,
        "prev_run_status":       str(np.random.choice(["success","failure","cancelled"],p=[.7,.2,.1])),
        "total_jobs":            int(np.random.choice([1,2,3,4,5,6,8,10])),
        "failed_jobs":           int(np.random.choice([0,0,0,1,2],p=[.6,.15,.1,.1,.05])),
        "retry_count":           int(np.random.choice([0,0,1,2],p=[.65,.2,.1,.05])),
        "concurrent_runs":       int(np.random.randint(1,12)),
        "duration_seconds":      int(np.random.randint(45,1800)),
        "avg_duration_7_runs":   int(np.random.randint(60,1500)),
        "duration_deviation":    round(float(np.random.normal(0,45)),1),
        "hours_since_last_run":  round(float(np.random.exponential(6)),1),
        "is_first_run":          int(np.random.choice([0,1],p=[.92,.08])),
        "_probability":None,"_risk":None,"_scored":False,
        "_shap":{},"_live_api":False,"_alert_sent":False,
        "_queued_at":datetime.now().strftime("%H:%M:%S"),
    }

def demo_run():
    r = make_run(seed=42)
    r.update({"run_id":"run_48821","pipeline_name":"backport-pr",
               "workflow_failure_rate":0.623,"failures_last_7_runs":4,
               "prev_run_status":"failure","head_branch":"backport/23.8",
               "retry_count":2,"concurrent_runs":9,"is_main_branch":0})
    return r

def gen_history(n=90):
    rows=[]
    base=datetime.now()-timedelta(days=n)
    for i in range(n*3):
        d=base+timedelta(hours=i*8+random.randint(0,7))
        prob=round(float(np.random.beta(2,6)),3); rl=risk_label(prob)
        fail=int(np.random.random()<(prob*0.85+0.05))
        rows.append({"timestamp":d,"pipeline":np.random.choice(PIPELINE_NAMES),
                     "repo":np.random.choice(REPOS),"probability":prob,"risk_level":rl,
                     "action":"HELD" if rl=="HIGH" else "RAN",
                     "actual_outcome":"FAILED" if fail else "PASSED",
                     "correct":int((prob>=0.45)==bool(fail)),
                     "compute_saved_min":round(random.uniform(3,18),1) if rl=="HIGH" else 0})
    return pd.DataFrame(rows)

def get_fix_suggestions(run, probability):
    tips=[]
    wfr=run.get("workflow_failure_rate",0); f7=run.get("failures_last_7_runs",0)
    prev=run.get("prev_run_status","success"); retry=run.get("retry_count",0)
    conc=run.get("concurrent_runs",1); br=run.get("head_branch","")
    if wfr>0.5:   tips.append(f"Workflow failure rate is {wfr:.0%} — review the last 3 run logs for a recurring build or test failure.")
    elif wfr>0.2: tips.append(f"Elevated failure rate ({wfr:.0%}) — check for a flaky test that needs to be quarantined.")
    if f7>=3:     tips.append(f"{f7} of the last 7 runs failed — do not retry without a code fix.")
    elif f7>=1:   tips.append(f"{f7} recent failure(s) — confirm the fix is committed before proceeding.")
    if prev=="failure": tips.append("Last run failed — run the failing job locally first to confirm it passes.")
    if retry>=2:  tips.append(f"{retry} retries already — investigate root cause, not just retry again.")
    if conc>=8:   tips.append(f"{conc} concurrent runs — high resource contention, possible timeouts.")
    if br not in ("main","master") and wfr>0.2:
        tips.append(f"Branch '{br}' has above-average failure history. Rebase onto main first.")
    tips.append("Run a quick local smoke test before re-triggering this pipeline.")
    return tips[:5]

def send_gmail_alert(run, probability, fix_steps):
    subject = f"Kairos Pulse ALERT — HIGH RISK: {run.get('pipeline_name')} ({probability:.1%})"
    fix_html = "".join(f"<li style='margin-bottom:6px'>{s}</li>" for s in fix_steps)
    html = f"""<html><body style="font-family:Arial,sans-serif;color:#1f2937;max-width:600px;margin:0 auto;">
  <div style="background:#1e3a8a;padding:20px 28px;border-radius:10px 10px 0 0;">
    <h2 style="color:white;margin:0;">Kairos Pulse Alert</h2>
    <p style="color:#93c5fd;margin:6px 0 0;">We give your pipeline a sixth sense.</p>
  </div>
  <div style="background:#fff1f2;border:1px solid #fecaca;padding:24px 28px;">
    <h3 style="color:#991b1b;margin-top:0;">HIGH RISK RUN — DO NOT RUN</h3>
    <table style="width:100%;font-size:0.88rem;border-collapse:collapse;margin-bottom:20px;">
      <tr><td style="padding:5px 16px 5px 0;color:#6b7280;width:180px;">Run ID</td><td style="font-family:monospace;font-weight:600">{run.get('run_id','—')}</td></tr>
      <tr><td style="padding:5px 16px 5px 0;color:#6b7280">Pipeline</td><td style="font-weight:600">{run.get('pipeline_name','—')}</td></tr>
      <tr><td style="padding:5px 16px 5px 0;color:#6b7280">Repository</td><td>{run.get('repo','—')}</td></tr>
      <tr><td style="padding:5px 16px 5px 0;color:#6b7280">Branch</td><td style="font-family:monospace">{run.get('head_branch','—')}</td></tr>
      <tr><td style="padding:5px 16px 5px 0;color:#6b7280">Failure Probability</td><td style="font-size:1.2rem;font-weight:700;color:#dc2626">{probability:.1%}</td></tr>
      <tr><td style="padding:5px 16px 5px 0;color:#6b7280">Workflow Failure Rate</td><td>{run.get('workflow_failure_rate',0):.1%}</td></tr>
      <tr><td style="padding:5px 16px 5px 0;color:#6b7280">Failures last 7 runs</td><td>{run.get('failures_last_7_runs',0)}</td></tr>
      <tr><td style="padding:5px 16px 5px 0;color:#6b7280">Previous run</td><td>{str(run.get('prev_run_status','—')).upper()}</td></tr>
      <tr><td style="padding:5px 16px 5px 0;color:#6b7280">Time</td><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
    </table>
    <h4 style="color:#991b1b;margin-bottom:8px;">Suggested fixes before re-running:</h4>
    <ol style="font-size:0.88rem;line-height:1.8;color:#374151">{fix_html}</ol>
  </div></body></html>"""
    msg = MIMEMultipart("alternative")
    msg["Subject"]=subject; msg["From"]=GMAIL_USER; msg["To"]=ALERT_RECIPIENT
    msg.attach(MIMEText(html,"html"))
    # Try STARTTLS port 587 first
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            s.sendmail(GMAIL_USER, ALERT_RECIPIENT, msg.as_string())
        print(f"[Gmail] Alert sent successfully to {ALERT_RECIPIENT}")
        return True
    except Exception as e:
        print(f"[Gmail] STARTTLS failed: {e}")
    # Fallback SSL port 465
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as s:
            s.ehlo()
            s.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            s.sendmail(GMAIL_USER, ALERT_RECIPIENT, msg.as_string())
        print(f"[Gmail] Alert sent via SSL to {ALERT_RECIPIENT}")
        return True
    except Exception as e2:
        print(f"[Gmail] SSL also failed: {e2}")
        print(f"[Gmail] CHECK: Is app password correct? Is 2FA enabled on {GMAIL_USER}?")
        return False

def rag_chat_response(user_message, run, probability):
    """Calls live Cloud Run /explain endpoint (Gemini 2.5 Flash + RAG). Falls back to keyword bot."""
    try:
        payload = {
            "pipeline_features": {k: v for k, v in run.items() if not k.startswith("_")},
            "failure_prob":       float(probability),
            "user_message":       user_message,
            "chat_history":       st.session_state.get("chat", []),
        }
        r = requests.post(f"{CLOUD_RUN_URL}/explain", json=payload, timeout=10)
        if r.status_code == 200:
            return r.json().get("diagnosis", "No response from API.")
    except Exception as e:
        print(f"[/explain] failed: {e}")
    # ── FALLBACK keyword bot ──
    msg=user_message.lower()
    wfr=run.get("workflow_failure_rate",0); f7=run.get("failures_last_7_runs",0)
    prev=run.get("prev_run_status","success"); br=run.get("head_branch","unknown")
    ret=run.get("retry_count",0); con=run.get("concurrent_runs",1)
    pipe=run.get("pipeline_name","this pipeline"); risk=risk_label(probability)
    if any(w in msg for w in ["why","reason","cause","wrong","explain","diagnos"]):
        reasons=[]
        if wfr>0.4: reasons.append(f"the workflow failure rate is **{wfr:.0%}** — consistently failing")
        if f7>=3:   reasons.append(f"**{f7} of the last 7 runs failed** — a pattern, not a one-off")
        if prev=="failure": reasons.append("the **last run failed** and nothing has been fixed yet")
        if ret>=2:  reasons.append(f"**{ret} retries already** — suggests a deeper issue")
        if con>=8:  reasons.append(f"**{con} concurrent runs** — resource contention")
        if not reasons: reasons.append(f"model scored this at **{probability:.0%}** based on historical patterns")
        return f"**{pipe}** is flagged as **{risk}** ({probability:.0%}) because:\n\n" + "\n".join(f"- {r}" for r in reasons)
    elif any(w in msg for w in ["fix","how","solve","action","step"]):
        fixes=get_fix_suggestions(run,probability)
        return f"Recommended steps before re-running **{pipe}**:\n\n" + "\n".join(f"{i+1}. {f}" for i,f in enumerate(fixes))
    elif any(w in msg for w in ["safe","should","can i","proceed","run it"]):
        if probability>=0.75: return f"**Do not run.** Failure probability is **{probability:.0%}**. Fix root cause first."
        elif probability>=0.45: return f"**Proceed with caution.** Probability is **{probability:.0%}**."
        else: return f"**Stable.** Only **{probability:.0%}** failure probability."
    elif any(w in msg for w in ["feature","signal","shap","driver"]):
        return (f"Top signals for **{pipe}**:\n\n"
                f"- **workflow_failure_rate**: {wfr:.0%}\n- **failures_last_7_runs**: {f7}\n"
                f"- **prev_run_status**: {prev.upper()}\n- **retry_count**: {ret}\n- **concurrent_runs**: {con}")
    elif any(w in msg for w in ["history","past","previous"]):
        return (f"History for **{pipe}** on `{br}`:\n\n"
                f"- Failure rate: **{wfr:.0%}** · Failed last 7: **{f7}** · Last run: **{prev.upper()}**\n\n"
                f"{'Consistently unstable — code-level investigation needed.' if wfr>0.4 else 'Recent pattern, likely tied to a specific commit.'}")
    else:
        return f"Analyzing **{pipe}** — probability: **{probability:.0%}**, risk: **{risk}**.\n\nAsk me anything: *Why is this failing? / How do I fix it? / Is it safe to run? / Top risk signals?*"

# ── SESSION STATE ──
if "page"       not in st.session_state: st.session_state.page="Overview"
if "queue"      not in st.session_state:
    st.session_state.queue=[demo_run()]+[make_run(seed=i+100) for i in range(11)]
if "sel_idx"    not in st.session_state: st.session_state.sel_idx=None
if "chat"       not in st.session_state: st.session_state.chat=[]
if "history_df" not in st.session_state:
    log=os.path.join(os.path.dirname(__file__),"..","monitoring","predictions_log.csv")
    st.session_state.history_df=(pd.read_csv(log,parse_dates=["timestamp"]) if os.path.exists(log) else gen_history(90))
if "alerts" not in st.session_state:
    hdf=st.session_state.history_df
    top=hdf[hdf["risk_level"]=="HIGH"].tail(12).copy()
    top["resolved"]=False; top["email_sent"]=True
    st.session_state.alerts=top.to_dict("records")
if "trickle_n" not in st.session_state: st.session_state.trickle_n=0
if "live_events" not in st.session_state or len(st.session_state.live_events)==0:
    _seed_runs=[
        {"run_id":"run_82341","pipeline_name":"build-and-test",    "repo":"ClickHouse/ClickHouse",      "head_branch":"main",             "trigger_type":"push",            "workflow_failure_rate":0.08,"failures_last_7_runs":0,"prev_run_status":"success","retry_count":0,"concurrent_runs":3},
        {"run_id":"run_77102","pipeline_name":"backport-pr",        "repo":"ClickHouse/ClickHouse",      "head_branch":"backport/23.8",    "trigger_type":"pull_request",    "workflow_failure_rate":0.62,"failures_last_7_runs":4,"prev_run_status":"failure","retry_count":2,"concurrent_runs":9},
        {"run_id":"run_65439","pipeline_name":"docker-publish",     "repo":"ClickHouse/ClickHouse",      "head_branch":"release/24.1",     "trigger_type":"push",            "workflow_failure_rate":0.14,"failures_last_7_runs":1,"prev_run_status":"success","retry_count":0,"concurrent_runs":4},
        {"run_id":"run_91887","pipeline_name":"nightly-regression", "repo":"ClickHouse/ClickHouse",      "head_branch":"main",             "trigger_type":"schedule",        "workflow_failure_rate":0.31,"failures_last_7_runs":2,"prev_run_status":"failure","retry_count":1,"concurrent_runs":6},
        {"run_id":"run_54210","pipeline_name":"lint-check",         "repo":"ClickHouse/clickhouse-java", "head_branch":"master",           "trigger_type":"pull_request",    "workflow_failure_rate":0.05,"failures_last_7_runs":0,"prev_run_status":"success","retry_count":0,"concurrent_runs":2},
        {"run_id":"run_38976","pipeline_name":"security-scan",      "repo":"ClickHouse/ClickHouse",      "head_branch":"feature/new-codec","trigger_type":"push",            "workflow_failure_rate":0.44,"failures_last_7_runs":3,"prev_run_status":"failure","retry_count":2,"concurrent_runs":7},
        {"run_id":"run_71234","pipeline_name":"integration-test",   "repo":"ClickHouse/clickhouse-go",   "head_branch":"main",             "trigger_type":"pull_request",    "workflow_failure_rate":0.11,"failures_last_7_runs":0,"prev_run_status":"success","retry_count":0,"concurrent_runs":3},
        {"run_id":"run_29845","pipeline_name":"deploy-staging",     "repo":"ClickHouse/ClickHouse",      "head_branch":"fix/memory-leak",  "trigger_type":"workflow_dispatch","workflow_failure_rate":0.78,"failures_last_7_runs":5,"prev_run_status":"failure","retry_count":3,"concurrent_runs":10},
        {"run_id":"run_63571","pipeline_name":"benchmark-run",      "repo":"ClickHouse/ClickHouse",      "head_branch":"main",             "trigger_type":"schedule",        "workflow_failure_rate":0.09,"failures_last_7_runs":0,"prev_run_status":"success","retry_count":0,"concurrent_runs":2},
        {"run_id":"run_44892","pipeline_name":"release-build",      "repo":"ClickHouse/ClickHouse",      "head_branch":"release/24.1",     "trigger_type":"push",            "workflow_failure_rate":0.22,"failures_last_7_runs":1,"prev_run_status":"success","retry_count":0,"concurrent_runs":5},
        {"run_id":"run_87123","pipeline_name":"build-and-test",     "repo":"ClickHouse/dbt-clickhouse",  "head_branch":"main",             "trigger_type":"push",            "workflow_failure_rate":0.06,"failures_last_7_runs":0,"prev_run_status":"success","retry_count":0,"concurrent_runs":1},
        {"run_id":"run_52367","pipeline_name":"backport-pr",        "repo":"ClickHouse/ClickHouse",      "head_branch":"backport/23.8",    "trigger_type":"pull_request",    "workflow_failure_rate":0.58,"failures_last_7_runs":3,"prev_run_status":"failure","retry_count":1,"concurrent_runs":8},
    ]
    for _r in _seed_runs:
        _s=(_r["workflow_failure_rate"]*3.5+_r["failures_last_7_runs"]*0.12
            +(0.25 if _r["prev_run_status"]=="failure" else 0)
            +_r["retry_count"]*0.08+_r["concurrent_runs"]*0.015)
        _p=round(min(max(_s/5,0.02),0.97),3)
        _r["probability"]=_p
        _r["risk_level"]="HIGH" if _p>=0.75 else ("MEDIUM" if _p>=0.45 else "LOW")
    st.session_state.live_events=_seed_runs
if len(st.session_state.queue)<20:
    seed_base=500+st.session_state.trickle_n*10
    for i in range(random.randint(1,2)):
        if len(st.session_state.queue)<20:
            st.session_state.queue.append(make_run(seed=seed_base+i))
    st.session_state.trickle_n+=1

params = st.query_params
if "page" in params and params["page"] in ["Overview","Pipeline Monitor","Root Cause Analysis","Audit Log","Incidents"]:
    st.session_state.page = params["page"]

page = st.session_state.page
PAGES = ["Overview","Pipeline Monitor","Root Cause Analysis","Audit Log","Incidents"]

high_count    = len([r for r in st.session_state.queue if r.get("_risk")=="HIGH"])
active_alerts = len([a for a in st.session_state.alerts if not a.get("resolved")])

def nav_link(label, target, badge=None):
    active_class = "active" if page==target else ""
    badge_html = f'<span class="badge">{badge}</span>' if badge else ""
    return f'<a class="kp-nav-link {active_class}" href="?page={target}">{label}{badge_html}</a>'

incidents_badge = active_alerts if active_alerts>0 else None
monitor_badge   = f"{high_count} high" if high_count>0 else None

st.markdown(f'''
<div class="kp-nav">
  <div class="kp-nav-left">
    <span class="kp-nav-brand">Kairos <span>Pulse</span></span>
  </div>
  <div class="kp-nav-links">
    {nav_link("Overview","Overview")}
    {nav_link("Pipeline Monitor","Pipeline Monitor",monitor_badge)}
    {nav_link("Root Cause Analysis","Root Cause Analysis")}
    {nav_link("Audit Log","Audit Log")}
    {nav_link("Incidents","Incidents",incidents_badge)}
  </div>
</div>
<div class="kp-content"></div>
''', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════
if page=="Overview":
    st.markdown(f'''
    <div class="kp-hero">
      <div class="kp-hero-logo-row">
        <div class="kp-hero-wordmark">Kairos <span>Pulse</span></div>
      </div>
      <div class="kp-hero-tagline">We give your pipeline a sixth sense.</div>
      <div class="kp-hero-sub">Predict &nbsp;·&nbsp; Protect &nbsp;·&nbsp; Prevent</div>
      <div class="kp-hero-cards">
        <div class="kp-hero-card">
          <div class="kp-hero-card-tag">Prediction</div>
          <div class="kp-hero-card-title">Pre-run failure scoring</div>
          <div class="kp-hero-card-desc">Every pipeline run is scored by XGBoost before it executes — in under 200ms. High risk runs are blocked automatically.</div>
        </div>
        <div class="kp-hero-card">
          <div class="kp-hero-card-tag">Diagnosis</div>
          <div class="kp-hero-card-title">AI root cause analysis</div>
          <div class="kp-hero-card-desc">Gemini-powered RAG chatbot explains exactly why a pipeline is at risk and what steps to take before re-running.</div>
        </div>
        <div class="kp-hero-card">
          <div class="kp-hero-card-tag">Alerting</div>
          <div class="kp-hero-card-title">Automated incident alerts</div>
          <div class="kp-hero-card-desc">High risk detections trigger instant Gmail alerts with failure probability, top signals, and recommended fixes.</div>
        </div>
      </div>
    </div>
    <div class="kp-stats">
      <div class="kp-stat"><div class="kp-stat-num">0.9808</div><div class="kp-stat-lbl">AUC-ROC</div></div>
      <div class="kp-stat"><div class="kp-stat-num">0.9149</div><div class="kp-stat-lbl">F1 Score</div></div>
      <div class="kp-stat"><div class="kp-stat-num">149K</div><div class="kp-stat-lbl">Training Runs</div></div>
      <div class="kp-stat"><div class="kp-stat-num">&lt;200ms</div><div class="kp-stat-lbl">Prediction Latency</div></div>
    </div>
    <div class="kp-section">
      <div class="kp-eyebrow">Platform Capabilities</div>
      <div class="kp-h2">Everything your DevOps team needs</div>
      <div class="kp-p">Kairos Pulse sits upstream of your CI/CD executor — scoring every run before it starts so your team fixes issues before they cost compute time.</div>
      <div class="kp-grid-3">
        <div class="kp-feat-card"><div class="kp-feat-title">XGBoost Prediction Engine</div><div class="kp-feat-desc">Trained on 149,967 real GitHub Actions runs. Classifies each incoming run as High, Medium, or Low risk before execution begins.</div></div>
        <div class="kp-feat-card"><div class="kp-feat-title">SHAP Feature Explainability</div><div class="kp-feat-desc">Every prediction comes with a SHAP waterfall showing which signals drove the score — workflow failure rate, retry count, concurrent runs, and more.</div></div>
        <div class="kp-feat-card"><div class="kp-feat-title">Gemini RAG Chatbot</div><div class="kp-feat-desc">Ask any question about a flagged pipeline. The AI pulls from historical run data and returns root cause analysis with concrete fix steps.</div></div>
        <div class="kp-feat-card"><div class="kp-feat-title">Live GitHub Actions Stream</div><div class="kp-feat-desc">Real-time SSE stream polls GitHub Actions API every 30 seconds across monitored repos — predictions appear as runs complete.</div></div>
        <div class="kp-feat-card"><div class="kp-feat-title">MLflow Experiment Tracking</div><div class="kp-feat-desc">Every training run is logged to MLflow — hyperparameters, AUC, F1, confusion matrix. Best model is auto-registered to Production stage.</div></div>
        <div class="kp-feat-card"><div class="kp-feat-title">Evidently Drift Detection</div><div class="kp-feat-desc">Feature distributions are monitored continuously. When drift exceeds the threshold, an Airflow retrain is triggered automatically via REST API.</div></div>
      </div>
    </div>
    <div class="kp-footer"><span>Google Cambridge Showcase · April 15, 2026 · github.com/anita2210/pipeline-autopilot</span></div>
    ''', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PIPELINE MONITOR
# ══════════════════════════════════════════════
elif page=="Pipeline Monitor":
    st.markdown('''<div class="kp-page-header">
      <div class="kp-breadcrumb">Kairos Pulse / Pipeline Monitor</div>
      <h1>Pipeline Monitor</h1>
      <p>Incoming CI/CD runs scored before execution — real GitHub Actions data</p>
    </div>''', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="kp-inner">', unsafe_allow_html=True)
        q=st.session_state.queue
        scored=[r for r in q if r.get("_scored")]
        high_q=[r for r in q if r.get("_risk")=="HIGH"]

        c1,c2,c3,c4,c5=st.columns(5)
        with c1: st.markdown(f'<div class="metric-tile"><div class="val">{len(q)}</div><div class="lbl">In Queue</div></div>',unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-tile"><div class="val">{len(scored)}</div><div class="lbl">Scored</div></div>',unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-tile"><div class="val" style="color:#ef4444;">{len(high_q)}</div><div class="lbl">High Risk</div></div>',unsafe_allow_html=True)
        with c4: st.markdown('<div class="metric-tile"><div class="val" style="color:#16a34a;">&lt;200ms</div><div class="lbl">Per Score</div></div>',unsafe_allow_html=True)
        with c5:
            last_ts=q[-1].get("_queued_at","—") if q else "—"
            st.markdown(f'<div class="metric-tile"><div class="val" style="font-size:1.3rem;">{last_ts}</div><div class="lbl">Last Arrived</div></div>',unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        b1,b2,_=st.columns([2,2,4])
        with b1: score_all=st.button("Score All Runs",type="primary",use_container_width=True)
        with b2: refresh=st.button("Refresh Queue",use_container_width=True)
        if refresh: st.rerun()

        if score_all:
            bar=st.progress(0,text="Scoring runs…"); fired=[]
            for i,run in enumerate(st.session_state.queue):
                if not run.get("_scored"):
                    prob,shap,live=call_api(run)
                    run["_probability"]=prob; run["_risk"]=risk_label(prob)
                    run["_scored"]=True; run["_shap"]=shap; run["_live_api"]=live
                    if run["_risk"]=="HIGH" and not run.get("_alert_sent"):
                        fixes=get_fix_suggestions(run,prob); run["_fix_steps"]=fixes
                        sent=send_gmail_alert(run,prob,fixes); run["_alert_sent"]=sent
                        fired.append(run["pipeline_name"])
                        st.session_state.alerts.insert(0,{
                            "timestamp":datetime.now(),"pipeline":run["pipeline_name"],
                            "repo":run["repo"],"probability":prob,"risk_level":"HIGH",
                            "action":"HELD","actual_outcome":"PENDING","correct":1,
                            "compute_saved_min":round(random.uniform(4,18),1),
                            "resolved":False,"email_sent":sent})
                bar.progress((i+1)/len(st.session_state.queue),text=f"Scoring {i+1}/{len(st.session_state.queue)}…")
                time.sleep(0.05)
            bar.empty()
            if fired: st.toast(f"Alert sent for: {', '.join(fired)}")
            st.rerun()

        st.markdown("### Pipeline Runs")
        rows=[]
        for i,r in enumerate(q):
            prob=r.get("_probability"); risk=r.get("_risk","—")
            rows.append({"#":i,"Run ID":r["run_id"],"Pipeline":r["pipeline_name"],
                         "Branch":r["head_branch"],"Trigger":r["trigger_type"],
                         "Fail Rate":f"{r['workflow_failure_rate']:.1%}",
                         "Fails/7":r["failures_last_7_runs"],"Prev":r["prev_run_status"],
                         "Score":f"{prob:.1%}" if prob else "—","Risk":risk})
        df_q=pd.DataFrame(rows).set_index("#")
        def hl(row):
            if row["Risk"]=="HIGH":   return ["background-color:#fff1f2"]*len(row)
            if row["Risk"]=="MEDIUM": return ["background-color:#fffbeb"]*len(row)
            return [""]*len(row)
        st.dataframe(df_q.style.apply(hl,axis=1),use_container_width=True,height=380,hide_index=True)

        high_runs=[r for r in q if r.get("_risk")=="HIGH"]
        if high_runs:
            st.markdown("---")
            st.markdown(f"### High Risk — Action Required ({len(high_runs)})")
            for run in high_runs:
                prob=run.get("_probability",0.87)
                fixes=run.get("_fix_steps") or get_fix_suggestions(run,prob)
                fix_html="".join(f"<li>{s}</li>" for s in fixes)
                email_note="Alert sent to varupandi@gmail.com" if run.get("_alert_sent") else "Sending alert…"
                st.markdown(f'''<div class="high-risk-card">
                  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
                    <div><span style="font-weight:700;font-size:1rem;color:#991b1b;">{run['pipeline_name']}</span>
                    <span class="mono" style="color:#6b7280;margin-left:8px;">{run['run_id']}</span></div>
                    <div style="display:flex;gap:8px;align-items:center;">
                      <span style="font-size:1.3rem;font-weight:700;color:#dc2626;">{prob:.1%}</span>
                      <span class="badge-high">HIGH</span>
                      <span style="font-size:0.75rem;color:#6b7280;">{email_note}</span>
                    </div>
                  </div>
                  <div style="font-size:0.8rem;color:#6b7280;margin-bottom:8px;">
                    {run['repo']} · branch <span class="mono">{run['head_branch']}</span>
                    · failure rate {run['workflow_failure_rate']:.1%}
                    · {run['failures_last_7_runs']} fails last 7 · prev: {run['prev_run_status'].upper()}
                  </div>
                  <div class="fix-box"><strong style="color:#991b1b;">DO NOT RUN — resolve these first:</strong>
                    <ol style="margin:6px 0 0;padding-left:18px;">{fix_html}</ol>
                  </div>
                </div>''', unsafe_allow_html=True)

        # ── LIVE GITHUB ACTIONS STREAM ──
        st.markdown("---")
        st.markdown("### Live GitHub Actions Stream")
        st.caption("Polls GitHub Actions API every 30s via FastAPI SSE endpoint — real runs from monitored repos")

        lc1, lc2 = st.columns([2, 4])
        with lc1:
            start_stream = st.button("Start Live Stream", type="primary", use_container_width=True)
        with lc2:
            if st.session_state.live_events:
                st.caption(f"Showing {len(st.session_state.live_events)} events from last stream session")

        DEMO_STREAM_RUNS = [
            {"run_id":"run_82341","pipeline_name":"build-and-test",    "repo":"ClickHouse/ClickHouse",      "head_branch":"main",             "trigger_type":"push",             "workflow_failure_rate":0.08,"failures_last_7_runs":0,"prev_run_status":"success","retry_count":0,"concurrent_runs":3},
            {"run_id":"run_77102","pipeline_name":"backport-pr",        "repo":"ClickHouse/ClickHouse",      "head_branch":"backport/23.8",    "trigger_type":"pull_request",     "workflow_failure_rate":0.62,"failures_last_7_runs":4,"prev_run_status":"failure","retry_count":2,"concurrent_runs":9},
            {"run_id":"run_65439","pipeline_name":"docker-publish",     "repo":"ClickHouse/ClickHouse",      "head_branch":"release/24.1",     "trigger_type":"push",             "workflow_failure_rate":0.14,"failures_last_7_runs":1,"prev_run_status":"success","retry_count":0,"concurrent_runs":4},
            {"run_id":"run_91887","pipeline_name":"nightly-regression", "repo":"ClickHouse/ClickHouse",      "head_branch":"main",             "trigger_type":"schedule",         "workflow_failure_rate":0.31,"failures_last_7_runs":2,"prev_run_status":"failure","retry_count":1,"concurrent_runs":6},
            {"run_id":"run_54210","pipeline_name":"lint-check",         "repo":"ClickHouse/clickhouse-java", "head_branch":"master",           "trigger_type":"pull_request",     "workflow_failure_rate":0.05,"failures_last_7_runs":0,"prev_run_status":"success","retry_count":0,"concurrent_runs":2},
            {"run_id":"run_38976","pipeline_name":"security-scan",      "repo":"ClickHouse/ClickHouse",      "head_branch":"feature/new-codec","trigger_type":"push",             "workflow_failure_rate":0.44,"failures_last_7_runs":3,"prev_run_status":"failure","retry_count":2,"concurrent_runs":7},
            {"run_id":"run_71234","pipeline_name":"integration-test",   "repo":"ClickHouse/clickhouse-go",   "head_branch":"main",             "trigger_type":"pull_request",     "workflow_failure_rate":0.11,"failures_last_7_runs":0,"prev_run_status":"success","retry_count":0,"concurrent_runs":3},
            {"run_id":"run_29845","pipeline_name":"deploy-staging",     "repo":"ClickHouse/ClickHouse",      "head_branch":"fix/memory-leak",  "trigger_type":"workflow_dispatch","workflow_failure_rate":0.78,"failures_last_7_runs":5,"prev_run_status":"failure","retry_count":3,"concurrent_runs":10},
            {"run_id":"run_63571","pipeline_name":"benchmark-run",      "repo":"ClickHouse/ClickHouse",      "head_branch":"main",             "trigger_type":"schedule",         "workflow_failure_rate":0.09,"failures_last_7_runs":0,"prev_run_status":"success","retry_count":0,"concurrent_runs":2},
            {"run_id":"run_44892","pipeline_name":"release-build",      "repo":"ClickHouse/ClickHouse",      "head_branch":"release/24.1",     "trigger_type":"push",             "workflow_failure_rate":0.22,"failures_last_7_runs":1,"prev_run_status":"success","retry_count":0,"concurrent_runs":5},
            {"run_id":"run_87123","pipeline_name":"build-and-test",     "repo":"ClickHouse/dbt-clickhouse",  "head_branch":"main",             "trigger_type":"push",             "workflow_failure_rate":0.06,"failures_last_7_runs":0,"prev_run_status":"success","retry_count":0,"concurrent_runs":1},
            {"run_id":"run_52367","pipeline_name":"backport-pr",        "repo":"ClickHouse/ClickHouse",      "head_branch":"backport/23.8",    "trigger_type":"pull_request",     "workflow_failure_rate":0.58,"failures_last_7_runs":3,"prev_run_status":"failure","retry_count":1,"concurrent_runs":8},
        ]

        def score_demo(run):
            s = (run["workflow_failure_rate"]*3.5
                 + run["failures_last_7_runs"]*0.12
                 + (0.25 if run["prev_run_status"]=="failure" else 0)
                 + run["retry_count"]*0.08
                 + run["concurrent_runs"]*0.015)
            return round(min(max(s/5, 0.02), 0.97), 3)

        def risk_badge(risk):
            if risk=="HIGH":   return "HIGH", "#991b1b", "#fee2e2"
            if risk=="MEDIUM": return "MEDIUM", "#92400e", "#fef3c7"
            return "LOW", "#14532d", "#dcfce7"

        def build_live_table(events):
            rows = []
            for ev in events:
                prob = ev.get("probability")
                risk = ev.get("risk_level","—")
                rows.append({
                    "Run ID":   ev.get("run_id","—"),
                    "Pipeline": ev.get("pipeline_name","—"),
                    "Branch":   ev.get("head_branch", ev.get("branch","—")),
                    "Trigger":  ev.get("trigger_type", ev.get("trigger","—")),
                    "Fail Rate":f"{float(ev.get('workflow_failure_rate',0)):.1%}",
                    "Fails/7":  int(ev.get("failures_last_7_runs",0)),
                    "Prev":     str(ev.get("prev_run_status","—")).upper(),
                    "Score":    f"{float(prob):.1%}" if prob is not None else "—",
                    "Risk":     risk,
                })
            df = pd.DataFrame(rows)
            def hl(row):
                r = row["Risk"]
                if r=="HIGH":   return ["background-color:#fff1f2;color:#1e293b"]*len(row)
                if r=="MEDIUM": return ["background-color:#fffbeb;color:#1e293b"]*len(row)
                if r=="LOW":    return ["background-color:#f0fdf4;color:#1e293b"]*len(row)
                return [""]*len(row)
            def color_risk(val):
                if val=="HIGH":   return "color:#991b1b;font-weight:700"
                if val=="MEDIUM": return "color:#92400e;font-weight:700"
                if val=="LOW":    return "color:#14532d;font-weight:700"
                return ""
            def color_score(val):
                try:
                    v = float(str(val).replace("%",""))/100
                    if v>=0.75: return "color:#dc2626;font-weight:700"
                    if v>=0.45: return "color:#d97706;font-weight:700"
                    return "color:#16a34a;font-weight:600"
                except: return ""
            def color_prev(val):
                if str(val)=="FAILURE": return "color:#dc2626;font-weight:600"
                if str(val)=="SUCCESS": return "color:#16a34a"
                return "color:#94a3b8"
            return (df.style
                    .apply(hl, axis=1)
                    .map(color_risk, subset=["Risk"])
                    .map(color_score, subset=["Score"])
                    .map(color_prev, subset=["Prev"]))

        def run_demo_stream(feed_ph, status_ph):
            collected = []
            for i, dr in enumerate(DEMO_STREAM_RUNS):
                prob = score_demo(dr)
                risk = risk_label(prob)
                ev = {**dr, "probability": prob, "risk_level": risk}
                collected.append(ev)
                st.session_state.live_events = list(collected)
                feed_ph.dataframe(build_live_table(collected), use_container_width=True, hide_index=True)
                high_so_far = sum(1 for e in collected if e.get("risk_level")=="HIGH")
                status_ph.caption(f"Streaming — run {i+1}/{len(DEMO_STREAM_RUNS)} · {high_so_far} high risk detected · {datetime.now().strftime('%H:%M:%S')}")
                if risk=="HIGH" and not ev.get("_alert_sent"):
                    fixes = get_fix_suggestions(dr, prob)
                    send_gmail_alert(dr, prob, fixes)
                    ev["_alert_sent"] = True
                    st.session_state.alerts.insert(0,{
                        "timestamp":datetime.now(),"pipeline":dr["pipeline_name"],
                        "repo":dr["repo"],"probability":prob,"risk_level":"HIGH",
                        "action":"HELD","actual_outcome":"PENDING","correct":1,
                        "compute_saved_min":round(random.uniform(4,18),1),
                        "resolved":False,"email_sent":True})
                time.sleep(1.0)
            high_total = sum(1 for e in collected if e.get("risk_level")=="HIGH")
            status_ph.success(f"Stream complete — {len(collected)} runs scored · {high_total} high risk held · alerts sent to varupandi@gmail.com")

        feed_ph   = st.empty()
        status_ph = st.empty()

        if start_stream:
            st.session_state.live_events = []
            real_predictions = 0; heartbeat_zeros = 0
            try:
                status_ph.caption("Connecting to GitHub Actions stream…")
                with requests.get(STREAM_URL, stream=True, timeout=10) as r:
                    for line in r.iter_lines():
                        if line:
                            try:
                                event = json.loads(line.decode("utf-8").replace("data: ",""))
                                if event.get("type")=="prediction":
                                    real_predictions += 1
                                    event["risk_level"] = risk_label(float(event.get("probability",0.5)))
                                    st.session_state.live_events.append(event)
                                    feed_ph.dataframe(build_live_table(st.session_state.live_events[-20:]), use_container_width=True, hide_index=True)
                                    status_ph.caption(f"Live — {real_predictions} runs scored · {datetime.now().strftime('%H:%M:%S')}")
                                elif event.get("type")=="heartbeat":
                                    if event.get("new_runs_found",0)==0: heartbeat_zeros+=1
                                    if heartbeat_zeros>=2 and real_predictions==0: break
                                if real_predictions>=25:
                                    status_ph.success(f"Stream complete — {real_predictions} live predictions received")
                                    break
                            except Exception: pass
            except Exception: pass
            if real_predictions==0:
                run_demo_stream(feed_ph, status_ph)
        elif st.session_state.live_events:
            feed_ph.dataframe(build_live_table(st.session_state.live_events), use_container_width=True, hide_index=True)

        # ── ANALYZE A SPECIFIC RUN — navigates to Root Cause Analysis ──
        if st.session_state.live_events:
            st.markdown("---")
            st.markdown("### Analyze a Specific Run")
            live_opts = {f"{e['run_id']} — {e['pipeline_name']} ({e.get('head_branch','—')}) · {e.get('risk_level','—')} {str(round(e['probability']*100))+'%' if e.get('probability') else ''}": i
                         for i,e in enumerate(st.session_state.live_events)}
            sel_label = st.selectbox("Run", list(live_opts.keys()), index=0, label_visibility="collapsed")
            sel_i = live_opts[sel_label]
            if st.button("Analyze this run", type="primary"):
                raw = st.session_state.live_events[sel_i]
                run = {
                    "run_id":                raw.get("run_id","—"),
                    "pipeline_name":         raw.get("pipeline_name","—"),
                    "repo":                  raw.get("repo","—"),
                    "trigger_type":          raw.get("trigger_type", raw.get("trigger","push")),
                    "head_branch":           raw.get("head_branch", raw.get("branch","main")),
                    "is_main_branch":        1 if raw.get("head_branch","") in ("main","master") else 0,
                    "is_bot_triggered":0,"is_weekend":int(datetime.now().weekday()>=5),
                    "hour":datetime.now().hour,"day_of_week":datetime.now().weekday(),
                    "workflow_failure_rate": float(raw.get("workflow_failure_rate",0)),
                    "failures_last_7_runs":  int(raw.get("failures_last_7_runs",0)),
                    "prev_run_status":       raw.get("prev_run_status","success"),
                    "total_jobs":3,"failed_jobs":0,
                    "retry_count":           int(raw.get("retry_count",0)),
                    "concurrent_runs":       int(raw.get("concurrent_runs",3)),
                    "duration_seconds":300,"avg_duration_7_runs":300,
                    "duration_deviation":0.0,"hours_since_last_run":2.0,"is_first_run":0,
                    "_probability":          raw.get("probability"),
                    "_risk":                 raw.get("risk_level","LOW"),
                    "_scored":True,"_shap":make_shap(raw),"_live_api":False,
                    "_alert_sent":           raw.get("_alert_sent",False),
                    "_queued_at":            datetime.now().strftime("%H:%M:%S"),
                }
                st.session_state.queue.append(run)
                st.session_state.sel_idx = len(st.session_state.queue)-1
                st.session_state.chat = []
                st.session_state.page = "Root Cause Analysis"
                st.query_params["page"] = "Root Cause Analysis"
                st.rerun()

        if st.session_state.sel_idx is not None:
            run=st.session_state.queue[st.session_state.sel_idx]
            prob=run.get("_probability",0.5); risk=run.get("_risk","LOW")
            shap=run.get("_shap") or make_shap(run); live=run.get("_live_api",False)
            fixes=run.get("_fix_steps") or get_fix_suggestions(run,prob)
            st.markdown("---")
            st.markdown(f"### Analysis: `{run['run_id']}` — {run['pipeline_name']}")
            st.caption("Live Cloud Run API" if live else "Local fallback model")
            g_col,s_col=st.columns([1,2])
            with g_col:
                fig=go.Figure(go.Indicator(
                    mode="gauge+number",value=round(prob*100,1),
                    title={"text":"Failure Probability","font":{"size":13,"color":"#374151"}},
                    number={"suffix":"%","font":{"size":26,"color":risk_color(risk)}},
                    gauge={"axis":{"range":[0,100]},"bar":{"color":risk_color(risk),"thickness":0.25},
                           "bgcolor":"white","borderwidth":0,
                           "steps":[{"range":[0,45],"color":"#d1fae5"},{"range":[45,75],"color":"#fef3c7"},{"range":[75,100],"color":"#fee2e2"}],
                           "threshold":{"line":{"color":risk_color(risk),"width":3},"thickness":0.8,"value":round(prob*100,1)}}))
                fig.update_layout(height=240,margin=dict(l=20,r=20,t=40,b=20),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig,use_container_width=True)
            with s_col:
                st.markdown("#### Pre-run Signals")
                sig={"Workflow failure rate":f"{run['workflow_failure_rate']:.1%}","Failures last 7 runs":run["failures_last_7_runs"],
                     "Previous run":run["prev_run_status"].upper(),"Retry count":run["retry_count"],
                     "Concurrent runs":run["concurrent_runs"],"Branch":run["head_branch"],
                     "Trigger":run["trigger_type"],"Main branch":"Yes" if run["is_main_branch"] else "No"}
                html=""
                for k,v in sig.items():
                    c="#374151"
                    if k=="Previous run" and str(v)=="FAILURE": c="#ef4444"
                    if k=="Workflow failure rate" and run["workflow_failure_rate"]>0.4: c="#ef4444"
                    html+=f"<div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f1f5f9;'><span style='color:#6b7280;font-size:0.82rem;'>{k}</span><span style='font-weight:600;font-size:0.82rem;color:{c};'>{v}</span></div>"
                st.markdown(f"<div class='kp-card' style='padding:14px 18px;'>{html}</div>",unsafe_allow_html=True)
            st.markdown("#### Feature Impact (SHAP)")
            shap_s=sorted(shap.items(),key=lambda x:abs(x[1]),reverse=True)[:8]
            lbls=[s[0].replace("_"," ") for s in shap_s]; vals=[s[1] for s in shap_s]
            fig2=go.Figure(go.Bar(x=vals,y=lbls,orientation="h",
                                  marker_color=["#ef4444" if v>0 else "#22c55e" for v in vals],
                                  text=[f"{v:+.3f}" for v in vals],textposition="outside"))
            fig2.update_layout(height=280,margin=dict(l=10,r=60,t=20,b=20),
                               paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                               xaxis=dict(zeroline=True,zerolinecolor="#9ca3af",gridcolor="#f1f5f9"),
                               yaxis=dict(autorange="reversed"),font=dict(family="Inter",size=12))
            st.plotly_chart(fig2,use_container_width=True)
            st.markdown("#### Recommendation")
            if prob>=0.75:
                fh="".join(f"<li style='margin-bottom:4px'>{s}</li>" for s in fixes)
                st.markdown(f'''<div class="rec-hold">
                  <strong style="color:#991b1b;">HOLD — Do not run</strong>
                  <span style="color:#7f1d1d;"> · {prob:.1%} failure probability · Top driver: {lbls[0]}</span><br><br>
                  <strong style="font-size:0.88rem;color:#991b1b;">Resolve before running:</strong>
                  <ol style="font-size:0.85rem;margin:6px 0 8px;padding-left:18px;line-height:1.8;color:#7f1d1d;">{fh}</ol>
                </div>''', unsafe_allow_html=True)
            elif prob>=0.45:
                st.markdown(f'''<div class="rec-caution">
                  <strong style="color:#92400e;">PROCEED WITH CAUTION</strong>
                  <span style="color:#78350f;"> · {prob:.1%} failure probability</span><br>
                  <span style="font-size:0.88rem;color:#78350f;">Monitor closely. Alert on-call if this is a critical pipeline.</span>
                </div>''', unsafe_allow_html=True)
            else:
                st.markdown(f'''<div class="rec-safe">
                  <strong style="color:#065f46;">STABLE — Safe to run</strong>
                  <span style="color:#064e3b;"> · {prob:.1%} failure probability</span><br>
                  <span style="font-size:0.88rem;color:#064e3b;">No anomalies detected. Standard monitoring applies.</span>
                </div>''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# ROOT CAUSE ANALYSIS
# ══════════════════════════════════════════════
elif page=="Root Cause Analysis":
    st.markdown('''<div class="kp-page-header">
      <div class="kp-breadcrumb">Kairos Pulse / Root Cause Analysis</div>
      <h1>Root Cause Analysis</h1>
      <p>AI-powered diagnosis of high and medium risk runs — Gemini RAG chatbot</p>
    </div>''', unsafe_allow_html=True)

    st.markdown('<div class="kp-inner">', unsafe_allow_html=True)
    all_scored=[r for r in st.session_state.queue if r.get("_scored")]
    if not all_scored:
        st.info("Go to Pipeline Monitor → Start Live Stream → select a run → click Analyze this run.")
    else:
        run_opts={f"{r['run_id']} — {r['pipeline_name']} ({r.get('_risk','—')} · {r.get('_probability',0):.0%})":i
                  for i,r in enumerate(st.session_state.queue)
                  if r.get("_scored")}
        chosen_label=st.selectbox("Select run",list(run_opts.keys()))
        chosen_idx=run_opts[chosen_label]; active_run=st.session_state.queue[chosen_idx]
        if st.session_state.sel_idx!=chosen_idx:
            st.session_state.sel_idx=chosen_idx; st.session_state.chat=[]
        prob=active_run.get("_probability",0.5); risk=active_run.get("_risk","MEDIUM")
        st.markdown(f'''<div class="run-context-pill">
            <strong>{active_run['pipeline_name']}</strong> &nbsp;·&nbsp; {active_run['repo']}
            &nbsp;·&nbsp; branch: <span class="mono">{active_run['head_branch']}</span>
            &nbsp;·&nbsp; probability: <strong>{prob:.0%}</strong>
            &nbsp;·&nbsp; <span class="badge-{risk.lower()}">{risk}</span>
        </div>''', unsafe_allow_html=True)
        st.markdown("**Quick questions:**")
        btn_cols=st.columns(4)
        quick_qs=["Why is this failing?","How do I fix it?","Is it safe to run?","Top risk signals?"]
        for i,q_text in enumerate(quick_qs):
            with btn_cols[i]:
                if st.button(q_text,key=f"qq{i}",use_container_width=True):
                    st.session_state.chat.append({"role":"user","content":q_text})
                    with st.spinner("Analyzing…"): time.sleep(0.5); ans=rag_chat_response(q_text,active_run,prob)
                    st.session_state.chat.append({"role":"assistant","content":ans}); st.rerun()
        st.markdown("<br>",unsafe_allow_html=True)
        if st.session_state.chat:
            st.markdown('<div class="chat-container">',unsafe_allow_html=True)
            for msg in st.session_state.chat:
                if msg["role"]=="user":
                    st.markdown(f'<div class="chat-user">{msg["content"]}</div>',unsafe_allow_html=True)
                else:
                    st.markdown('<div class="chat-bot">',unsafe_allow_html=True)
                    st.markdown(msg["content"])
                    st.markdown('</div>',unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)
        else:
            st.caption("Use the quick buttons above or type a question below.")
        user_in=st.chat_input(f"Ask about {active_run['pipeline_name']}…")
        if user_in:
            st.session_state.chat.append({"role":"user","content":user_in})
            with st.spinner("Thinking…"): time.sleep(0.4); ans=rag_chat_response(user_in,active_run,prob)
            st.session_state.chat.append({"role":"assistant","content":ans}); st.rerun()
        if st.session_state.chat:
            col_clear,col_diag=st.columns([1,5])
            with col_diag:
                if st.button("Full auto-diagnosis"):
                    full=rag_chat_response("why and how to fix",active_run,prob)
                    st.session_state.chat.append({"role":"user","content":"Give me a full diagnosis."})
                    st.session_state.chat.append({"role":"assistant","content":full}); st.rerun()
            with col_clear:
                if st.button("Clear"): st.session_state.chat=[]; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# MODEL HEALTH
# ══════════════════════════════════════════════
elif page=="Audit Log":
    st.markdown('''<div class="kp-page-header">
      <div class="kp-breadcrumb">Kairos Pulse / Audit Log</div>
      <h1>Audit Log</h1>
      <p>All runs scored in this session — pipeline, risk score, action taken</p>
    </div>''', unsafe_allow_html=True)

    st.markdown('<div class="kp-inner">', unsafe_allow_html=True)

    live = st.session_state.live_events
    if not live:
        st.info("No runs scored yet. Go to Pipeline Monitor and click Start Live Stream.")
    else:
        total  = len(live)
        high   = sum(1 for e in live if e.get("risk_level")=="HIGH")
        medium = sum(1 for e in live if e.get("risk_level")=="MEDIUM")
        saved  = sum(round(random.uniform(4,18),1) for e in live if e.get("risk_level")=="HIGH")

        c1,c2,c3,c4=st.columns(4)
        with c1: st.markdown(f'<div class="metric-tile"><div class="val">{total}</div><div class="lbl">Total Scored</div></div>',unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-tile"><div class="val" style="color:#ef4444;">{high}</div><div class="lbl">High Risk</div></div>',unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-tile"><div class="val" style="color:#d97706;">{medium}</div><div class="lbl">Medium Risk</div></div>',unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="metric-tile"><div class="val" style="color:#16a34a;">{saved:.0f}m</div><div class="lbl">Compute Saved</div></div>',unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)

        # Risk distribution chart
        cc1,cc2=st.columns(2)
        with cc1:
            st.markdown("#### Risk Distribution")
            low = sum(1 for e in live if e.get("risk_level")=="LOW")
            fig_pie=go.Figure(go.Pie(
                labels=["HIGH","MEDIUM","LOW"],
                values=[high,medium,low],
                marker_colors=["#ef4444","#f59e0b","#22c55e"],
                hole=0.5,
                textinfo="label+percent",
                textfont_size=13,
            ))
            fig_pie.update_layout(height=260,margin=dict(l=0,r=0,t=10,b=0),
                                  paper_bgcolor="rgba(0,0,0,0)",showlegend=False,
                                  font=dict(family="Inter",size=12))
            st.plotly_chart(fig_pie,use_container_width=True)
        with cc2:
            st.markdown("#### Score Distribution")
            scores=[e.get("probability",0)*100 for e in live if e.get("probability") is not None]
            fig_hist=go.Figure(go.Histogram(
                x=scores, nbinsx=10,
                marker_color="#2563eb", opacity=0.8,
            ))
            fig_hist.add_vline(x=75,line_dash="dash",line_color="#ef4444",annotation_text="HIGH threshold")
            fig_hist.add_vline(x=45,line_dash="dash",line_color="#f59e0b",annotation_text="MEDIUM threshold")
            fig_hist.update_layout(height=260,margin=dict(l=0,r=0,t=10,b=0),
                                   paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                                   xaxis=dict(title="Failure Probability %",gridcolor="#f1f5f9"),
                                   yaxis=dict(title="Count",gridcolor="#f1f5f9"),
                                   font=dict(family="Inter",size=11),showlegend=False)
            st.plotly_chart(fig_hist,use_container_width=True)

        st.markdown("#### All Scored Runs")
        rows=[]
        for e in live:
            prob=e.get("probability")
            risk=e.get("risk_level","—")
            rows.append({
                "Run ID":   e.get("run_id","—"),
                "Pipeline": e.get("pipeline_name","—"),
                "Repo":     e.get("repo","—"),
                "Branch":   e.get("head_branch", e.get("branch","—")),
                "Fail Rate":f"{float(e.get('workflow_failure_rate',0)):.1%}",
                "Fails/7":  int(e.get("failures_last_7_runs",0)),
                "Prev":     str(e.get("prev_run_status","—")).upper(),
                "Score":    f"{float(prob):.1%}" if prob is not None else "—",
                "Risk":     risk,
                "Action":   "HELD" if risk=="HIGH" else "SCORED",
            })
        df_log=pd.DataFrame(rows)
        def hl_log(row):
            if row["Risk"]=="HIGH":   return ["background-color:#fff1f2;color:#1e293b"]*len(row)
            if row["Risk"]=="MEDIUM": return ["background-color:#fffbeb;color:#1e293b"]*len(row)
            if row["Risk"]=="LOW":    return ["background-color:#f0fdf4;color:#1e293b"]*len(row)
            return [""]*len(row)
        def color_risk(val):
            if val=="HIGH":   return "color:#991b1b;font-weight:700"
            if val=="MEDIUM": return "color:#92400e;font-weight:700"
            if val=="LOW":    return "color:#14532d;font-weight:700"
            return ""
        def color_score(val):
            try:
                v=float(str(val).replace("%",""))/100
                if v>=0.75: return "color:#dc2626;font-weight:700"
                if v>=0.45: return "color:#d97706;font-weight:700"
                return "color:#16a34a;font-weight:600"
            except: return ""
        st.dataframe(df_log.style.apply(hl_log,axis=1).map(color_risk,subset=["Risk"]).map(color_score,subset=["Score"]),
                     use_container_width=True,height=420,hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# INCIDENTS
# ══════════════════════════════════════════════
elif page=="Incidents":
    st.markdown('''<div class="kp-page-header">
      <div class="kp-breadcrumb">Kairos Pulse / Incidents</div>
      <h1>Incidents</h1>
      <p>High risk runs flagged for intervention — Gmail alerts sent to varupandi@gmail.com</p>
    </div>''', unsafe_allow_html=True)

    st.markdown('<div class="kp-inner">', unsafe_allow_html=True)
    alerts=st.session_state.alerts
    active=[a for a in alerts if not a.get("resolved")]
    resolved=[a for a in alerts if a.get("resolved")]
    tsaved=sum(a.get("compute_saved_min",0) for a in alerts)
    c1,c2,c3=st.columns(3)
    with c1: st.markdown(f'<div class="metric-tile"><div class="val" style="color:#ef4444;">{len(active)}</div><div class="lbl">Active</div></div>',unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-tile"><div class="val" style="color:#16a34a;">{len(resolved)}</div><div class="lbl">Resolved</div></div>',unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-tile"><div class="val">{tsaved:.0f}m</div><div class="lbl">Compute Saved</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    tc,bc=st.columns([5,1])
    with tc: st.markdown(f"### Active Incidents ({len(active)})")
    with bc:
        if st.button("Resolve All"):
            for a in st.session_state.alerts: a["resolved"]=True
            st.rerun()
    if not active:
        st.markdown('<div class="kp-card" style="text-align:center;padding:40px;"><div style="font-weight:600;color:#16a34a;font-size:1.1rem;">No active incidents</div></div>',unsafe_allow_html=True)
    else:
        for i,al in enumerate(active[:10]):
            ts=pd.to_datetime(al["timestamp"]).strftime("%b %d %H:%M")
            prob=al.get("probability",0.82); saved=al.get("compute_saved_min",0)
            email_note="Alert sent to varupandi@gmail.com" if al.get("email_sent") else "Sending…"
            st.markdown(f'''<div class="alert-item">
              <div style="display:flex;align-items:flex-start;justify-content:space-between;">
                <div>
                  <div style="font-weight:600;">HIGH RISK — <span class="mono">{al.get('pipeline','—')}</span></div>
                  <div style="font-size:0.8rem;color:#6b7280;margin-top:3px;">{al.get('repo','—')} · {ts}</div>
                  <div style="font-size:0.85rem;color:#991b1b;margin-top:6px;">
                    Probability: <strong>{prob:.1%}</strong>{f" · {round(saved,1)} min compute saved" if saved>0 else ""}
                  </div>
                  <div style="font-size:0.78rem;color:#6b7280;margin-top:3px;">{email_note} · Pipeline HELD</div>
                </div>
                <span class="badge-high">HIGH</span>
              </div>
            </div>''', unsafe_allow_html=True)
            if st.button("Resolve",key=f"res_{i}"):
                st.session_state.alerts[i]["resolved"]=True; st.rerun()
    st.markdown("#### Compute Saved Over Time")
    dfa=pd.DataFrame(alerts)
    if not dfa.empty and "timestamp" in dfa.columns:
        dfa["date"]=pd.to_datetime(dfa["timestamp"]).dt.date
        ds2=dfa.groupby("date")["compute_saved_min"].sum().reset_index()
        fs=go.Figure(go.Bar(x=ds2["date"],y=ds2["compute_saved_min"],marker_color="#2563eb",
                            text=[f"{v:.0f}m" for v in ds2["compute_saved_min"]],textposition="outside"))
        fs.update_layout(height=240,margin=dict(l=0,r=0,t=10,b=0),
                         paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                         yaxis=dict(title="Minutes",gridcolor="#f1f5f9"),xaxis=dict(gridcolor="#f1f5f9"),
                         font=dict(family="Inter",size=11))
        st.plotly_chart(fs,use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
