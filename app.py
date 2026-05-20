import streamlit as st
import os, sys, uuid, joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
import arabic_reshaper
from bidi.algorithm import get_display

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from src.nlp.nlp_processor import EgyptianStudentNLP
from src.recommendation.recommendation_engine import RecommendationEngine
from src.conversation.conversation_manager import ConversationManager, ConversationContext

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Scholara — Academic Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# STYLES — Bootstrap 5 + Material Symbols + Custom Design System
# ══════════════════════════════════════════════════════════════════════════════
def inject_styles():
    raw_css = """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,1,0" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,300;400;500;600;700;800&family=Tajawal:wght@300;400;500;700;800;900&display=swap" rel="stylesheet">
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── TOKENS ────────────────────────────────────────────────────────────── */
    :root {
        --canvas:     #05070f;
        --surface:    #0b0e1c;
        --elevated:   #111526;
        --lift:       #182038;
        --hover:      #1e2640;

        --border:     rgba(120,140,255,0.09);
        --border-md:  rgba(120,140,255,0.16);
        --border-hi:  rgba(120,140,255,0.28);

        --teal:       #00dfc4;
        --teal-dim:   rgba(0,223,196,0.12);
        --teal-glow:  rgba(0,223,196,0.25);
        --blue:       #4f8fff;
        --blue-dim:   rgba(79,143,255,0.12);
        --violet:     #b06ef5;
        --violet-dim: rgba(176,110,245,0.12);
        --amber:      #f4a22d;
        --amber-dim:  rgba(244,162,45,0.12);
        --green:      #12c98a;
        --green-dim:  rgba(18,201,138,0.12);
        --red:        #f2415a;
        --red-dim:    rgba(242,65,90,0.12);

        --text:       #edf0ff;
        --text-2:     #8b92bc;
        --text-3:     #444d78;

        --r-xs: 6px;  --r-sm: 10px;  --r-md: 16px;
        --r-lg: 22px; --r-xl: 30px;

        --font: 'Inter', 'Bricolage Grotesque', 'Tajawal', sans-serif;

        --transition: 0.18s cubic-bezier(0.4,0,0.2,1);
        --glass: rgba(255,255,255,0.03);
        --glass-border: rgba(255,255,255,0.06);
    }

    /* ── BASE ──────────────────────────────────────────────────────────────── */
    html, body, [class*="css"], .stApp, p, span, div, label {
        font-family: var(--font);
        color: var(--text);
    }
    /* Restores Streamlit's default icon fonts and expander toggle icons */
    [data-testid="collapsedControl"] svg, [data-testid="stSidebarCollapseButton"] svg, [data-testid="stExpanderToggleIcon"] {
        font-family: 'Material Symbols Rounded', sans-serif !important;
        font-size: 24px !important;
    }
    @keyframes cosmic-breath {
        0% {
            background-position: 0% 10%, 100% 90%, 50% 40%;
            opacity: 0.92;
        }
        25% {
            background-position: 10% 20%, 90% 80%, 60% 50%;
            opacity: 1.0;
        }
        50% {
            background-position: 20% 0%, 80% 100%, 40% 60%;
            opacity: 0.88;
        }
        75% {
            background-position: 10% 15%, 90% 85%, 55% 45%;
            opacity: 1.0;
        }
        100% {
            background-position: 0% 10%, 100% 90%, 50% 40%;
            opacity: 0.92;
        }
    }
    .stApp {
        background-color: var(--canvas) !important;
        background-image:
            radial-gradient(circle 800px at 5% 15%,  rgba(0,223,196,0.065) 0%, transparent 80%),
            radial-gradient(circle 900px at 95% 85%, rgba(176,110,245,0.075) 0%, transparent 80%),
            radial-gradient(circle 700px at 50% 50%, rgba(79,143,255,0.055) 0%, transparent 70%);
        background-attachment: fixed !important;
        background-size: 200% 200% !important;
        animation: cosmic-breath 14s ease-in-out infinite alternate !important;
    }
    header[data-testid="stHeader"], footer, #MainMenu { display: none !important; }

    /* ── SIDEBAR ───────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #05070f 0%, #080b18 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

    /* ── TYPOGRAPHY ────────────────────────────────────────────────────────── */
    h1,h2,h3,h4,h5 {
        font-family: var(--font) !important;
        color: var(--text) !important;
        letter-spacing: -0.025em;
        font-weight: 700 !important;
    }

    /* ── BUTTONS ───────────────────────────────────────────────────────────── */
    .stButton > button {
        font-family: var(--font) !important;
        background: var(--elevated) !important;
        color: var(--text-2) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--r-sm) !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        transition: all var(--transition) !important;
        letter-spacing: -0.01em;
    }
    .stButton > button:hover {
        background: var(--hover) !important;
        border-color: var(--border-md) !important;
        color: var(--text) !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.35);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, rgba(0,223,196,0.18), rgba(79,143,255,0.14)) !important;
        border: 1px solid rgba(0,223,196,0.35) !important;
        color: var(--teal) !important;
        font-weight: 700 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, rgba(0,223,196,0.26), rgba(79,143,255,0.2)) !important;
        border-color: rgba(0,223,196,0.55) !important;
        box-shadow: 0 0 28px rgba(0,223,196,0.18), 0 6px 20px rgba(0,0,0,0.3);
    }

    /* ── CHAT INPUT ────────────────────────────────────────────────────────── */
    [data-testid="stChatInput"] {
        background: var(--elevated) !important;
        border: 1px solid var(--border-md) !important;
        border-radius: var(--r-xl) !important;
        box-shadow: 0 4px 30px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04) !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(0,223,196,0.45) !important;
        box-shadow: 0 0 0 4px rgba(0,223,196,0.08), 0 4px 30px rgba(0,0,0,0.5) !important;
    }
    [data-testid="stChatInput"] textarea { color: var(--text) !important; font-family: var(--font) !important; }

    /* ── CHAT MESSAGES ─────────────────────────────────────────────────────── */
    [data-testid="stChatMessage"] { background: transparent !important; }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
        background: linear-gradient(135deg, rgba(0,223,196,0.1), rgba(79,143,255,0.07)) !important;
        border: 1px solid rgba(0,223,196,0.22) !important;
        border-radius: var(--r-md) var(--r-md) 4px var(--r-md) !important;
        padding: 14px 20px !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
        background: var(--elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--r-md) var(--r-md) var(--r-md) 4px !important;
        padding: 14px 20px !important;
    }

    /* ── FORM CONTROLS ─────────────────────────────────────────────────────── */
    input, textarea, select,
    [data-baseweb="input"] > div,
    [data-baseweb="select"] > div,
    .stSelectbox > div > div {
        background: var(--elevated) !important;
        color: var(--text) !important;
        border: 1px solid var(--border-md) !important;
        border-radius: var(--r-sm) !important;
        font-family: var(--font) !important;
    }
    input:focus, textarea:focus {
        border-color: rgba(0,223,196,0.45) !important;
        box-shadow: 0 0 0 3px rgba(0,223,196,0.1) !important;
    }
    [data-baseweb="select"] svg { color: var(--text-2) !important; }
    label { color: var(--text-2) !important; font-size: 0.82rem !important; font-weight: 600 !important; }

    /* Premium Sleek Number Inputs */
    div[data-testid="stNumberInput"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(120, 140, 255, 0.1) !important;
        border-radius: var(--r-md) !important;
        padding: 6px 12px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        margin-bottom: 12px !important;
    }
    div[data-testid="stNumberInput"]:hover {
        border-color: rgba(0, 223, 196, 0.3) !important;
        background: rgba(255, 255, 255, 0.04) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }
    div[data-testid="stNumberInput"]:focus-within {
        border-color: var(--teal) !important;
        box-shadow: 0 0 16px var(--teal-glow) !important;
        background: rgba(0, 223, 196, 0.02) !important;
    }
    /* Style the label */
    div[data-testid="stNumberInput"] label {
        color: var(--text-2) !important;
        font-size: 0.76rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        margin-bottom: 6px !important;
        display: block !important;
    }
    /* Style the input container */
    div[data-testid="stNumberInput"] > div:nth-child(2) {
        background: transparent !important;
        border: none !important;
        display: flex !important;
        align-items: center !important;
    }
    /* Style the actual input field */
    div[data-testid="stNumberInput"] input {
        background: rgba(0, 0, 0, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-size: 1.1rem !important;
        font-weight: 800 !important;
        text-align: center !important;
        height: 38px !important;
        transition: all 0.18s ease !important;
    }
    div[data-testid="stNumberInput"] input:focus {
        color: var(--teal) !important;
        border-color: rgba(0, 223, 196, 0.35) !important;
        background: rgba(0, 0, 0, 0.3) !important;
    }
    /* Style the controls buttons */
    div[data-testid="stNumberInput"] button {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        color: var(--text-2) !important;
        width: 36px !important;
        height: 36px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 1.1rem !important;
        font-weight: bold !important;
        transition: all 0.2s ease !important;
        margin: 0 2px !important;
    }
    div[data-testid="stNumberInput"] button:hover {
        background: linear-gradient(135deg, rgba(0, 223, 196, 0.2), rgba(79, 143, 255, 0.15)) !important;
        color: var(--teal) !important;
        border-color: rgba(0, 223, 196, 0.4) !important;
        transform: scale(1.05) !important;
    }
    div[data-testid="stNumberInput"] button:active {
        transform: scale(0.95) !important;
    }

    /* Premium Glassmorphic Sidebar Expander */
    div[data-testid="stExpander"] {
        background: linear-gradient(135deg, rgba(17,21,38,0.3) 0%, rgba(11,14,28,0.4) 100%) !important;
        border: 1px solid rgba(120, 140, 255, 0.07) !important;
        border-radius: var(--r-md) !important;
        overflow: hidden !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        margin-top: 10px !important;
    }
    div[data-testid="stExpander"] summary {
        background: transparent !important;
        padding: 10px 14px !important;
        font-weight: 600 !important;
        color: var(--text-2) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stExpander"] summary:hover {
        background: rgba(255, 255, 255, 0.015) !important;
        color: var(--text) !important;
    }

    /* ── SLIDERS ───────────────────────────────────────────────────────────── */
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background: var(--teal) !important;
        border-color: var(--teal) !important;
        box-shadow: 0 0 12px var(--teal-glow) !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stSliderTrackFill"] {
        background: linear-gradient(90deg, var(--teal), var(--blue)) !important;
    }

    /* ── METRICS ───────────────────────────────────────────────────────────── */
    [data-testid="stMetricValue"]  { color: var(--teal) !important; font-weight: 800 !important; }
    [data-testid="stMetricLabel"]  { color: var(--text-2) !important; font-size: 0.78rem !important; }
    [data-testid="stMetricDelta"]  { font-size: 0.8rem !important; }

    /* ── TABS ──────────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--r-md) !important;
        padding: 5px !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-2) !important;
        border-radius: var(--r-sm) !important;
        font-family: var(--font) !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        padding: 9px 20px !important;
        border: none !important;
        transition: all var(--transition) !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--elevated) !important;
        color: var(--text) !important;
        border: 1px solid var(--border-md) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }

    /* ── DATAFRAME ─────────────────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--r-md) !important;
        overflow: hidden !important;
    }

    /* ── SCROLLBAR ─────────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border-md); border-radius: 99px; }

    /* ── MATERIAL SYMBOLS ──────────────────────────────────────────────────── */
    .ms {
        font-family: 'Material Symbols Rounded' !important;
        font-weight: normal !important;
        font-style: normal !important;
        font-size: 20px !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-block !important;
        white-space: nowrap !important;
        direction: ltr !important;
        -webkit-font-smoothing: antialiased !important;
        vertical-align: middle !important;
    }

    /* ══════════════════════════════════════════════════════════════════════
       COMPONENT LIBRARY
    ══════════════════════════════════════════════════════════════════════ */

    /* Shimmering breath effect for cards */
    @keyframes card-glow {
        0%, 100% {
            box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04);
            border-color: var(--glass-border);
        }
        50% {
            box-shadow: 0 8px 30px rgba(0, 223, 196, 0.05), inset 0 1px 0 rgba(255,255,255,0.08);
            border-color: rgba(0, 223, 196, 0.15);
        }
    }
    .sc-card {
        background: linear-gradient(135deg, rgba(17,21,38,0.95) 0%, rgba(11,14,28,0.98) 100%);
        border: 1px solid var(--glass-border);
        border-radius: var(--r-lg);
        padding: 22px 24px;
        margin-bottom: 18px;
        position: relative;
        overflow: hidden;
        transition: all 0.25s cubic-bezier(0.4,0,0.2,1);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04);
        animation: card-glow 8s ease-in-out infinite alternate;
    }
    .sc-card::before {
        content: '';
        position: absolute; inset: 0;
        border-radius: inherit;
        background: linear-gradient(135deg, rgba(255,255,255,0.025) 0%, transparent 50%);
        pointer-events: none;
    }
    .sc-card:hover {
        border-color: rgba(120,140,255,0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
        transform: translateY(-1px);
    }

    /* Accent card variants */
    .sc-card.teal   { border-top: 2px solid var(--teal);   box-shadow: 0 4px 24px rgba(0,0,0,0.3), 0 0 0 0 rgba(0,223,196,0.1); }
    .sc-card.blue   { border-top: 2px solid var(--blue);   box-shadow: 0 4px 24px rgba(0,0,0,0.3), 0 0 0 0 rgba(79,143,255,0.1); }
    .sc-card.violet { border-top: 2px solid var(--violet); box-shadow: 0 4px 24px rgba(0,0,0,0.3), 0 0 0 0 rgba(176,110,245,0.1); }
    .sc-card.amber  { border-top: 2px solid var(--amber);  box-shadow: 0 4px 24px rgba(0,0,0,0.3), 0 0 0 0 rgba(244,162,45,0.1); }
    .sc-card.green  { border-top: 2px solid var(--green);  box-shadow: 0 4px 24px rgba(0,0,0,0.3), 0 0 0 0 rgba(18,201,138,0.1); }
    .sc-card.red    { border-top: 2px solid var(--red);    box-shadow: 0 4px 24px rgba(0,0,0,0.3), 0 0 0 0 rgba(242,65,90,0.1); }

    /* Brand Header */
    .brand-wrap {
        padding: 22px 18px 18px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 13px;
    }
    .brand-icon {
        width: 40px; height: 40px;
        background: linear-gradient(135deg, var(--teal), var(--blue));
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-family: var(--font) !important;
        font-size: 1.1rem; font-weight: 900;
        color: #05070f;
        flex-shrink: 0;
        box-shadow: 0 4px 16px rgba(0,223,196,0.3);
    }
    .brand-name  { font-size: 1.3rem; font-weight: 800; color: var(--text); letter-spacing: -0.04em; line-height: 1.1; }
    .brand-sub   { font-size: 0.65rem; color: var(--text-3); text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }

    /* Nav section */
    .nav-sec { font-size: 0.65rem; color: var(--text-3); text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700; padding: 14px 20px 6px; }

    /* Page header */
    .ph-wrap { margin-bottom: 28px; }
    .ph-title { display: flex; align-items: center; gap: 14px; margin-bottom: 4px; }
    .ph-icon {
        width: 44px; height: 44px;
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }
    .ph-h2 { font-size: 1.65rem !important; font-weight: 800 !important; margin: 0 !important; letter-spacing: -0.03em; }
    .ph-sub { color: var(--text-3); font-size: 0.84rem; margin-top: 2px; }

    /* Score bars */
    .sb-wrap { margin-bottom: 13px; }
    .sb-row   { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
    .sb-label { font-size: 0.82rem; color: var(--text-2); font-weight: 600; display: flex; align-items: center; gap: 6px; }
    .sb-val   { font-size: 0.82rem; font-weight: 700; }
    .sb-track { background: rgba(255,255,255,0.04); border-radius: 99px; height: 5px; overflow: hidden; border: 1px solid var(--border); }
    /* Score bar animation */
    @keyframes bar-grow {
        from { width: 0 !important; }
        to   { width: var(--bar-w); }
    }
    .sb-fill { animation: bar-grow 0.7s cubic-bezier(0.4,0,0.2,1) both; }

    /* Clickable Chips — rendered as streamlit buttons */
    .chip-grid { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; margin-top: 24px; }
    .chip-btn {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(120,140,255,0.12);
        border-radius: 99px;
        padding: 10px 18px;
        font-size: 0.84rem;
        color: var(--text-2);
        cursor: pointer;
        font-weight: 600;
        transition: all 0.2s cubic-bezier(0.4,0,0.2,1);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        white-space: nowrap;
    }
    .chip-btn:hover {
        border-color: rgba(0,223,196,0.4);
        color: var(--teal);
        background: rgba(0,223,196,0.08);
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,223,196,0.15);
    }
    /* Override streamlit button style for chips */
    .chip-container button {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(120,140,255,0.15) !important;
        border-radius: 99px !important;
        color: var(--text-2) !important;
        font-family: var(--font) !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        padding: 8px 18px !important;
        transition: all 0.2s ease !important;
        backdrop-filter: blur(8px) !important;
    }
    .chip-container button:hover {
        border-color: rgba(0,223,196,0.45) !important;
        color: var(--teal) !important;
        background: rgba(0,223,196,0.1) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,223,196,0.2) !important;
    }

    /* Stat pill */
    .spill {
        display: inline-flex; align-items: center; gap: 5px;
        background: var(--hover); border: 1px solid var(--border);
        border-radius: 99px; padding: 4px 11px;
        font-size: 0.77rem; color: var(--text-2); font-weight: 600;
    }
    .spill.t { border-color: rgba(0,223,196,.3);  color: var(--teal);   background: var(--teal-dim); }
    .spill.g { border-color: rgba(18,201,138,.3); color: var(--green);  background: var(--green-dim); }
    .spill.r { border-color: rgba(242,65,90,.3);  color: var(--red);    background: var(--red-dim); }
    .spill.a { border-color: rgba(244,162,45,.3); color: var(--amber);  background: var(--amber-dim); }
    .spill.b { border-color: rgba(79,143,255,.3); color: var(--blue);   background: var(--blue-dim); }

    /* Tip box */
    .tip-box {
        background: var(--teal-dim);
        border: 1px solid rgba(0,223,196,0.18);
        border-radius: var(--r-md);
        padding: 14px 18px;
        display: flex; gap: 12px; align-items: flex-start;
        margin-bottom: 16px;
    }
    .tip-box .ms { color: var(--teal) !important; font-size: 18px !important; margin-top: 1px; flex-shrink: 0; }
    .tip-box p  { margin: 0; font-size: 0.83rem; color: var(--text-2); line-height: 1.55; }

    /* Welcome screen */
    .welcome-wrap { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 56px 20px; text-align: center; }
    .welcome-orb {
        width: 76px; height: 76px;
        background: radial-gradient(circle, rgba(0,223,196,0.2) 0%, rgba(79,143,255,0.1) 100%);
        border: 1px solid rgba(0,223,196,0.25);
        border-radius: 22px;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 24px;
        animation: orb-pulse 3.5s ease-in-out infinite;
    }
    @keyframes orb-pulse {
        0%,100% { box-shadow: 0 0 0 0 rgba(0,223,196,0); }
        50%      { box-shadow: 0 0 0 14px rgba(0,223,196,0.07); }
    }
    .welcome-title {
        font-size: 2.1rem; font-weight: 800;
        background: linear-gradient(135deg, var(--teal) 0%, var(--blue) 50%, var(--violet) 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        letter-spacing: -0.04em; margin-bottom: 10px;
    }
    .welcome-sub { color: var(--text-3); font-size: 0.88rem; max-width: 360px; line-height: 1.65; }

    /* Chips */
    .chip {
        display: inline-flex; align-items: center; gap: 7px;
        background: var(--elevated); border: 1px solid var(--border);
        border-radius: 99px; padding: 7px 15px;
        font-size: 0.81rem; color: var(--text-2); cursor: pointer; margin: 4px;
        font-weight: 600; transition: all var(--transition);
    }
    .chip:hover { border-color: rgba(0,223,196,0.35); color: var(--teal); background: var(--teal-dim); }

    /* Result badge */
    .result-wrap {
        padding: 32px 24px; border-radius: var(--r-xl);
        text-align: center; position: relative; overflow: hidden;
        margin-bottom: 18px;
    }
    .result-wrap .big-num { font-size: 3.5rem; font-weight: 800; line-height: 1; margin-bottom: 4px; }
    .result-wrap .status-tag {
        font-size: 0.7rem; font-weight: 800;
        letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 16px;
    }

    /* Score input card (predictor) */
    .score-input-row {
        display: grid; grid-template-columns: 1fr 1fr 1fr;
        gap: 14px; margin-bottom: 10px;
    }
    .score-input-item {
        background: var(--lift);
        border: 1px solid var(--border);
        border-radius: var(--r-md);
        padding: 14px 16px;
        cursor: pointer;
        transition: all var(--transition);
        position: relative;
        overflow: hidden;
    }
    .score-input-item:hover { border-color: var(--border-md); background: var(--hover); }
    .score-input-item .sii-label { font-size: 0.75rem; color: var(--text-3); font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
    .score-input-item .sii-val   { font-size: 1.6rem; font-weight: 800; color: var(--text); line-height: 1; }
    .score-input-item .sii-max   { font-size: 0.75rem; color: var(--text-3); margin-top: 2px; }
    .score-input-item .sii-bar   { position: absolute; bottom: 0; left: 0; height: 3px; transition: width 0.5s ease; }

    /* Divider */
    .sc-hr { border: none; border-top: 1px solid var(--border); margin: 20px 0; }

    /* Number input arrows */
    input[type=number]::-webkit-inner-spin-button { opacity: 0.5; }

    /* Info section label */
    .sec-label {
        font-size: 0.68rem; font-weight: 800;
        text-transform: uppercase; letter-spacing: 0.12em;
        color: var(--text-3); margin-bottom: 14px;
        display: flex; align-items: center; gap: 8px;
    }
    .sec-label::after {
        content: ''; flex: 1; height: 1px;
        background: linear-gradient(90deg, var(--border), transparent);
    }

    /* Grade ring */
    .grade-ring-wrap { text-align: center; padding: 20px; }
    .grade-letter { font-size: 5rem; font-weight: 900; line-height: 1; letter-spacing: -0.05em; }
    .grade-ring-sub { font-size: 0.8rem; color: var(--text-3); margin-top: 4px; }

    /* Prediction stepper */
    .step-pill {
        display: inline-flex; align-items: center; gap: 8px;
        background: var(--lift); border: 1px solid var(--border);
        border-radius: 99px; padding: 5px 14px 5px 5px;
        font-size: 0.78rem; color: var(--text-2); font-weight: 600; margin-bottom: 16px;
    }
    .step-num {
        width: 22px; height: 22px; border-radius: 99px;
        background: linear-gradient(135deg, var(--teal), var(--blue));
        color: #05070f; font-size: 0.72rem; font-weight: 800;
        display: flex; align-items: center; justify-content: center;
    }

    /* Analytics big number cards */
    .ana-stat {
        background: var(--elevated); border: 1px solid var(--border);
        border-radius: var(--r-lg); padding: 20px;
        text-align: center; transition: all var(--transition);
        position: relative; overflow: hidden;
    }
    .ana-stat:hover { border-color: var(--border-md); transform: translateY(-2px); }
    .ana-stat .ana-icon { font-size: 2rem !important; margin-bottom: 10px; display: block; }
    .ana-stat .ana-val  { font-size: 1.9rem; font-weight: 900; letter-spacing: -0.04em; line-height: 1; }
    .ana-stat .ana-label{ font-size: 0.75rem; color: var(--text-3); margin-top: 5px; font-weight: 600; }

    /* Tips card */
    .tip-card {
        background: var(--elevated); border: 1px solid var(--border);
        border-radius: var(--r-lg); padding: 20px;
        margin-bottom: 16px; transition: all var(--transition);
    }
    .tip-card:hover { border-color: var(--border-md); transform: translateY(-1px); }
    .tip-card .tc-icon {
        width: 44px; height: 44px; border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        margin-bottom: 14px;
    }
    .tip-card .tc-title { font-size: 0.95rem; font-weight: 700; color: var(--text); margin-bottom: 7px; }
    .tip-card .tc-body  { font-size: 0.81rem; color: var(--text-2); line-height: 1.62; }

    /* Sidebar mini chart */
    .mini-scores { padding: 0 2px; }
    .ms-row { display: flex; align-items: center; gap: 8px; margin-bottom: 9px; }
    .ms-name { font-size: 0.72rem; color: var(--text-3); font-weight: 600; width: 60px; flex-shrink: 0; }
    .ms-bar-track { flex: 1; height: 4px; background: rgba(255,255,255,0.05); border-radius: 99px; overflow: hidden; }
    .ms-bar-fill  { height: 100%; border-radius: 99px; }
    .ms-num { font-size: 0.72rem; font-weight: 700; width: 30px; text-align: right; flex-shrink: 0; }

    </style>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
    """
    st.html(raw_css)

inject_styles()

# ══════════════════════════════════════════════════════════════════════════════
# RESOURCES
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Initializing Scholara...")
def load_resources():
    nlp     = EgyptianStudentNLP()
    rec     = RecommendationEngine()
    manager = ConversationManager(nlp, rec)
    base    = os.path.dirname(__file__)
    csv     = os.path.join(base, "data", "raw", "Students Performance Dataset.csv")
    df      = pd.read_csv(csv) if os.path.exists(csv) else None
    mdir    = os.path.join(base, "models")
    load    = lambda f: joblib.load(os.path.join(mdir, f)) if os.path.exists(os.path.join(mdir, f)) else None
    return manager, df, load("rf_model.pkl"), load("xgb_model.pkl"), load("feature_names.pkl"), load("model_metadata.pkl")

manager, df_dataset, rf_model, xgb_model, feature_names, metadata = load_resources()

# ══════════════════════════════════════════════════════════════════════════════
# SESSION DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════
SCORE_DEFAULTS = {"Midterm_Score": 0, "Final_Score": 0, "Assignments_Avg": 0,
                  "Quizzes_Avg": 0, "Projects_Score": 0, "Participation_Score": 0}

for k, v in {
    "lang": "ar",
    "nav_page": "chat",
    "manual_scores": SCORE_DEFAULTS.copy(),
    "chip_input": None,
}.items():
    if k not in st.session_state: st.session_state[k] = v

is_ar = st.session_state.lang == "ar"

# ══════════════════════════════════════════════════════════════════════════════
# TRANSLATIONS
# ══════════════════════════════════════════════════════════════════════════════
def t(ar, en): return ar if is_ar else en

def t_plot(ar, en):
    text = ar if is_ar else en
    if is_ar:
        return get_display(arabic_reshaper.reshape(text))
    return text

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
FEATURES = [
    ("Midterm_Score",       t("الميدترم","Midterm"),       30,  "quiz",              "#00dfc4"),
    ("Final_Score",         t("الفاينل","Final Exam"),     50,  "school",            "#4f8fff"),
    ("Assignments_Avg",     t("الواجبات","Assignments"),   100, "assignment",        "#b06ef5"),
    ("Quizzes_Avg",         t("الكويزات","Quizzes"),       100, "help_outline",      "#f4a22d"),
    ("Projects_Score",      t("المشاريع","Projects"),      100, "rocket_launch",     "#12c98a"),
    ("Participation_Score", t("المشاركة","Participation"), 10,  "record_voice_over", "#f2415a"),
]

DEPT_SUBJECTS = {
    "CS":          ["Data Structures","Algorithms","Operating Systems","Databases","Computer Networks"],
    "IT":          ["Networking","System Admin","Cybersecurity","Cloud Computing"],
    "IS":          ["Information Retrieval","Data Mining","Enterprise Systems","BI & Analytics"],
    "SWE":         ["Software Engineering","Testing & QA","Project Management","DevOps"],
    "AI":          ["Machine Learning","Deep Learning","NLP","Computer Vision","Reinforcement Learning"],
    "Mathematics": ["Calculus","Linear Algebra","Discrete Math","Probability & Statistics"],
    "Business":    ["Accounting","Marketing","Economics","Management"],
    "Engineering": ["Physics","Mechanics","Circuits","Thermodynamics"],
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _ensure_ctx(obj):
    if isinstance(obj, ConversationContext): return obj
    ctx = ConversationContext()
    if isinstance(obj, dict):
        ctx.state        = obj.get("state","idle")
        ctx.pending_score= obj.get("pending_score")
        ctx.known_scores = obj.get("known_scores",{})
        ctx.last_intent  = obj.get("last_intent")
        ctx.turn_count   = obj.get("turn_count",0)
    return ctx

def dark_fig(w=7, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#0b0e1c')
    ax.set_facecolor('#0b0e1c')
    for s in ax.spines.values(): s.set_color('#182038')
    ax.tick_params(colors='#444d78', labelsize=9)
    ax.xaxis.label.set_color('#8b92bc')
    ax.yaxis.label.set_color('#8b92bc')
    return fig, ax

def dark_figs(rows, cols, w=10, h=5):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor('#0b0e1c')
    for ax in np.array(axes).flatten():
        ax.set_facecolor('#0b0e1c')
        for s in ax.spines.values(): s.set_color('#182038')
        ax.tick_params(colors='#444d78', labelsize=8)
        ax.xaxis.label.set_color('#8b92bc')
        ax.yaxis.label.set_color('#8b92bc')
    return fig, axes

def score_css(ratio):
    if ratio >= 0.8: return "#12c98a", "var(--green)"
    if ratio >= 0.6: return "#00dfc4", "var(--teal)"
    if ratio >= 0.4: return "#f4a22d", "var(--amber)"
    return "#f2415a", "var(--red)"

def grade_letter(pct):
    if pct >= 90: return "A+", "#12c98a"
    if pct >= 85: return "A",  "#12c98a"
    if pct >= 80: return "B+", "#00dfc4"
    if pct >= 75: return "B",  "#00dfc4"
    if pct >= 70: return "C+", "#4f8fff"
    if pct >= 65: return "C",  "#4f8fff"
    if pct >= 60: return "D",  "#f4a22d"
    return "F", "#f2415a"

def section_label(txt, icon=""):
    st.markdown(f"""<div class="sec-label"><span class="ms" style="font-size:14px!important;color:var(--text-3);">{icon}</span>{txt}</div>""", unsafe_allow_html=True)

def card_open(cls=""):
    st.markdown(f'<div class="sc-card {cls}">', unsafe_allow_html=True)

def card_close():
    st.markdown('</div>', unsafe_allow_html=True)

def divider():
    st.markdown('<hr class="sc-hr">', unsafe_allow_html=True)

def score_bars_html(scores_dict, features):
    html = ""
    for col, label, mx, icon, color in features:
        val   = scores_dict.get(col, 0)
        ratio = val / mx if mx else 0
        hx, _ = score_css(ratio)
        html += f"""
        <div class="sb-wrap">
            <div class="sb-row">
                <span class="sb-label">
                    <span class="ms" style="font-size:15px!important;color:{hx}">{icon}</span>{label}
                </span>
                <span class="sb-val" style="color:{hx};">{val}<span style="color:var(--text-3);font-weight:400;">/{mx}</span></span>
            </div>
            <div class="sb-track"><div class="sb-fill" style="width:{ratio*100:.1f}%;background:linear-gradient(90deg,{hx}88,{hx});"></div></div>
        </div>"""
    return html

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="brand-wrap">
        <div class="brand-icon">S</div>
        <div>
            <div class="brand-name">Scholara</div>
            <div class="brand-sub">Academic Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Language toggle
    la, le = st.columns(2)
    with la:
        if st.button("العربية", use_container_width=True, type="primary" if is_ar else "secondary"):
            st.session_state.lang = "ar"; st.rerun()
    with le:
        if st.button("English", use_container_width=True, type="primary" if not is_ar else "secondary"):
            st.session_state.lang = "en"; st.rerun()

    st.markdown('<hr class="sc-hr">', unsafe_allow_html=True)

    # Navigation
    nav = [
        ("chat",       "forum",        t("المساعد الذكي","AI Advisor")),
        ("predictor",  "trending_up",  t("التنبؤ بالنتيجة","Grade Predictor")),
        ("calculator", "calculate",    t("حاسبة الدرجات","Score Calculator")),
        ("analytics",  "bar_chart",    t("الإحصائيات","Analytics")),
        ("tips",       "lightbulb",    t("نصائح الدراسة","Study Tips")),
    ]
    for pid, icon, label in nav:
        active = pid == st.session_state.nav_page
        if st.button(label, key=f"nav_{pid}", use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state.nav_page = pid; st.rerun()

    if st.session_state.nav_page == "chat":
        st.markdown('<hr class="sc-hr">', unsafe_allow_html=True)
        if st.button(t("محادثة جديدة","New Chat"), use_container_width=True):
            nid = str(uuid.uuid4())
            if "chats" not in st.session_state:
                st.session_state.chats = {}
            st.session_state.chats[nid] = {"title": t("محادثة جديدة","New Chat"), "messages": [], "ctx": ConversationContext()}
            st.session_state.current_chat = nid; st.rerun()
        st.markdown(f'<div class="nav-sec">{t("المحادثات","Chats")}</div>', unsafe_allow_html=True)
        if "chats" not in st.session_state:
            cid = str(uuid.uuid4())
            st.session_state.chats = {cid: {"title": t("محادثة جديدة","New Chat"), "messages": [], "ctx": ConversationContext()}}
            st.session_state.current_chat = cid
        for cid, cd in reversed(list(st.session_state.chats.items())):
            if st.button(cd["title"][:26], key=f"c_{cid}", use_container_width=True):
                st.session_state.current_chat = cid; st.rerun()

    # Sidebar mini progress
    st.markdown('<hr class="sc-hr">', unsafe_allow_html=True)
    st.markdown(f'<div class="nav-sec">{t("ملخص درجاتك","Your Scores")}</div>', unsafe_allow_html=True)
    mini_html = '<div class="mini-scores">'
    for col, label, mx, icon, color in FEATURES:
        val = st.session_state.manual_scores.get(col, 0)
        r   = val / mx if mx else 0
        mini_html += f"""
        <div class="ms-row">
            <div class="ms-name">{label[:6]}</div>
            <div class="ms-bar-track"><div class="ms-bar-fill" style="width:{r*100:.0f}%;background:{color};"></div></div>
            <div class="ms-num" style="color:{color};">{val}</div>
        </div>"""
    mini_html += '</div>'
    st.markdown(mini_html, unsafe_allow_html=True)

    # Removed Score Config Expander from here (moved to dedicated Settings page)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CHAT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.nav_page == "chat":
    if "chats" not in st.session_state:
        cid = str(uuid.uuid4())
        st.session_state.chats = {cid: {"title": t("محادثة جديدة","New Chat"), "messages": [], "ctx": ConversationContext()}}
        st.session_state.current_chat = cid
    chat_data = st.session_state.chats[st.session_state.current_chat]

    col_chat, col_panel = st.columns([1.8, 1], gap="large")

    with col_chat:
        st.markdown(f"""
        <div class="ph-wrap">
            <div class="ph-title">
                <div class="ph-icon" style="background:var(--teal-dim);border:1px solid rgba(0,223,196,0.2);">
                    <span class="ms" style="color:var(--teal);font-size:24px!important">forum</span>
                </div>
                <div>
                    <h2 class="ph-h2">{t("المساعد الأكاديمي الذكي","AI Academic Advisor")}</h2>
                    <div class="ph-sub">{t("تحدث معي عن درجاتك وخططك الأكاديمية","Talk to me about your grades and academic plans")}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not chat_data["messages"]:
            sug = [
                ("quiz",         t("هل درجاتي تكفي للنجاح؟","Will my grades be enough?"),   "#00dfc4"),
                ("menu_book",    t("اعملي خطة مذاكرة","Make me a study plan"),              "#4f8fff"),
                ("trending_up",  t("كيف أرفع معدلي؟","How to raise my GPA?"),               "#b06ef5"),
                ("psychology",   t("أنا قلقان من الامتحانات","I'm anxious about exams"),     "#f4a22d"),
                ("calculate",    t("احسبلي توتال سكور","Calculate my total score"),          "#12c98a"),
                ("school",       t("أي مادة أذاكر الأول؟","Which subject first?"),           "#f2415a"),
            ]
            # Render the welcome hero section
            st.markdown(f"""
            <div class="welcome-wrap">
                <div class="welcome-orb"><span class="ms" style="font-size:2.2rem!important;color:var(--teal)">school</span></div>
                <div class="welcome-title">{t("أهلاً بك في Scholara","Welcome to Scholara")}</div>
                <p class="welcome-sub">{t("مساعدك الأكاديمي الذكي. أخبرني عن درجاتك، خططك، أو اسألني أي سؤال أكاديمي.","Your intelligent academic advisor. Tell me your grades, plans, or ask any academic question.")}</p>
            </div>
            """, unsafe_allow_html=True)

            # Clickable chips using real st.buttons styled as chips
            st.markdown('<div class="chip-container">', unsafe_allow_html=True)
            chip_cols = st.columns(3)
            for i, (ic, txt, clr) in enumerate(sug):
                with chip_cols[i % 3]:
                    label_html = f'<span style="font-size:15px;vertical-align:middle;font-family:\"Material Symbols Rounded\";margin-right:5px;color:{clr}">{ic}</span>{txt}'
                    if st.button(txt, key=f"chip_{i}", use_container_width=True):
                        st.session_state.chip_input = txt
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            for msg in chat_data["messages"]:
                with st.chat_message(msg["role"], avatar="user" if msg["role"]=="user" else "assistant"):
                    st.markdown(msg["content"])

        user_input = st.chat_input(t("اسألني أي شيء...","Ask Scholara anything..."))

        # Handle chip button click
        if st.session_state.chip_input:
            user_input = st.session_state.chip_input
            st.session_state.chip_input = None

        if user_input and user_input.strip():
            raw = user_input.strip()
            if not chat_data["messages"]: chat_data["title"] = raw[:24] + "…"
            chat_data["ctx"] = _ensure_ctx(chat_data["ctx"])
            for col, *_ in FEATURES:
                if col in st.session_state.manual_scores:
                    chat_data["ctx"].known_scores[col] = st.session_state.manual_scores[col]
            with st.spinner(t("جاري التحليل...","Analyzing...")):
                response, user_display = manager.handle(
                    user_text=raw, ctx=chat_data["ctx"],
                    lang="العربية" if is_ar else "English"
                )
            for col, val in chat_data["ctx"].known_scores.items():
                st.session_state.manual_scores[col] = val
            chat_data["messages"].append({"role":"user","content":user_display})
            chat_data["messages"].append({"role":"assistant","content":response})
            st.rerun()

    with col_panel:
        st.markdown("<div style='margin-top:58px;'></div>", unsafe_allow_html=True)

        # Grade Tracker card
        card_open()
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">
            <span class="ms" style="color:var(--teal)">monitoring</span>
            <div>
                <div style="font-size:.95rem;font-weight:700;">{t("متتبع الدرجات","Grade Tracker")}</div>
                <div style="font-size:.74rem;color:var(--text-3);">{t("مزامنة تلقائية مع المحادثة","Auto-synced from chat")}</div>
            </div>
        </div>
        {score_bars_html(st.session_state.manual_scores, FEATURES)}
        """, unsafe_allow_html=True)
        card_close()

        # Live grade estimate (use static maxes)
        ms_  = st.session_state.manual_scores
        est  = (ms_["Midterm_Score"] / 30) * 30 + \
               (ms_["Assignments_Avg"] / 100) * 10 + \
               (ms_["Quizzes_Avg"] / 100) * 10
        gl, gc = grade_letter(est)
        card_open()
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;">
            <div>
                <div style="font-size:.68rem;color:var(--text-3);text-transform:uppercase;letter-spacing:.1em;font-weight:700;margin-bottom:4px;">{t("تقديرك الحالي","Current Estimate")}</div>
                <div class="grade-letter" style="color:{gc};">{gl}</div>
                <div class="grade-ring-sub">{est:.1f} / 100 {t("(بدون الفاينل)","(without final)")}</div>
            </div>
            <span class="ms" style="font-size:4rem!important;color:{gc};opacity:.18;">grade</span>
        </div>
        """, unsafe_allow_html=True)
        card_close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: GRADE PREDICTOR — All sliders, clear layout
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.nav_page == "predictor":
    _mx = {
        "Midterm_Score": 30,
        "Final_Score": 50,
        "Assignments_Avg": 100,
        "Quizzes_Avg": 100,
        "Projects_Score": 100,
        "Participation_Score": 10
    }
    st.markdown(f"""
    <div class="ph-wrap">
        <div class="ph-title">
            <div class="ph-icon" style="background:var(--blue-dim);border:1px solid rgba(79,143,255,0.22);">
                <span class="ms" style="color:var(--blue);font-size:24px!important">trending_up</span>
            </div>
            <div>
                <h2 class="ph-h2">{t("التنبؤ بنتيجة المادة","Grade Predictor")}</h2>
                <div class="ph-sub">{t("أدخل درجاتك بسهولة واحصل على تنبؤ فوري من نموذجين ذكيين","Easily enter your grades and get an instant dual-model AI prediction")}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if rf_model is None or xgb_model is None:
        st.error(t("النماذج غير موجودة — يرجى التدريب أولاً","Models not found — please train first."))
    else:
        left, right = st.columns([1.55, 1], gap="large")

        with left:
            # ─ Section 1: Department & Subject ─
            card_open()
            section_label(t("القسم والمادة","Department & Subject"), "school")
            dc, sc = st.columns(2)
            with dc:
                dept_in = st.selectbox(t("القسم","Department"), list(DEPT_SUBJECTS.keys()), label_visibility="collapsed")
            with sc:
                subj_in = st.selectbox(t("المادة","Subject"), DEPT_SUBJECTS[dept_in], label_visibility="collapsed")
            card_close()

            # ─ Section 2: Academic Scores — ALL SLIDERS ─
            card_open("teal")
            section_label(t("الدرجات الأكاديمية","Academic Scores"), "quiz")

            st.markdown(f"""<div class="tip-box">
                <span class="ms">info</span>
                <p>{t("حرّك الشريط لإدخال كل درجة — لا تحتاج للكتابة اليدوية","Drag each slider to enter your score — no typing needed")}</p>
            </div>""", unsafe_allow_html=True)

            mid_in  = st.slider(
                t(f"الميدترم (من {_mx['Midterm_Score']})", f"Midterm (out of {_mx['Midterm_Score']})"),
                0, int(_mx["Midterm_Score"]),
                min(int(st.session_state.manual_scores.get("Midterm_Score", 0)), int(_mx["Midterm_Score"]))
            ) if int(_mx["Midterm_Score"]) > 0 else 0
            fin_in  = st.slider(
                t(f"الفاينل (من {_mx['Final_Score']})", f"Final Exam (out of {_mx['Final_Score']})"),
                0, int(_mx["Final_Score"]),
                min(int(st.session_state.manual_scores.get("Final_Score", 0)), int(_mx["Final_Score"]))
            ) if int(_mx["Final_Score"]) > 0 else 0
            att_in  = st.slider(t("الحضور (%)","Attendance (%)"), 0, 100, 85)
            asgn_in = st.slider(
                t(f"الواجبات (من {_mx['Assignments_Avg']})", f"Assignments (out of {_mx['Assignments_Avg']})"),
                0, int(_mx["Assignments_Avg"]),
                min(int(st.session_state.manual_scores.get("Assignments_Avg", 0)), int(_mx["Assignments_Avg"]))
            ) if int(_mx["Assignments_Avg"]) > 0 else 0
            quiz_in = st.slider(
                t(f"الكويزات (من {_mx['Quizzes_Avg']})", f"Quizzes (out of {_mx['Quizzes_Avg']})"),
                0, int(_mx["Quizzes_Avg"]),
                min(int(st.session_state.manual_scores.get("Quizzes_Avg", 0)), int(_mx["Quizzes_Avg"]))
            ) if int(_mx["Quizzes_Avg"]) > 0 else 0
            proj_in = st.slider(
                t(f"المشاريع (من {_mx['Projects_Score']})", f"Projects (out of {_mx['Projects_Score']})"),
                0, int(_mx["Projects_Score"]),
                min(int(st.session_state.manual_scores.get("Projects_Score", 0)), int(_mx["Projects_Score"]))
            ) if int(_mx["Projects_Score"]) > 0 else 0
            part_in = st.slider(
                t(f"المشاركة (من {_mx['Participation_Score']})", f"Participation (out of {_mx['Participation_Score']})"),
                0, int(_mx["Participation_Score"]),
                min(int(st.session_state.manual_scores.get("Participation_Score", 0)), int(_mx["Participation_Score"]))
            ) if int(_mx["Participation_Score"]) > 0 else 0

            # Update session state
            for k, v in zip(
                ["Midterm_Score","Final_Score","Assignments_Avg","Quizzes_Avg","Projects_Score","Participation_Score"],
                [mid_in, fin_in, asgn_in, quiz_in, proj_in, part_in]
            ): st.session_state.manual_scores[k] = v

            card_close()

            # ─ Section 3: Personal Factors ─
            card_open("violet")
            section_label(t("عوامل شخصية","Personal Factors"), "person")

            p1, p2 = st.columns(2)
            with p1:
                gender_in = st.selectbox(t("الجنس","Gender"),         ["Male","Female"])
                age_in    = st.slider(t("العمر","Age"),                15, 40, 20)
                hours_in  = st.slider(t("ساعات دراسة/أسبوع","Study hrs/week"), 0, 60, 15)
            with p2:
                sleep_in  = st.slider(t("ساعات النوم","Sleep hours"),  4, 12, 7)
                stress_in = st.slider(t("مستوى التوتر (1-10)","Stress level (1-10)"), 1, 10, 5)
                income_in = st.selectbox(t("دخل الأسرة","Family income"),["Low","Medium","High"])

            q1, q2, q3 = st.columns(3)
            with q1: parent_in  = st.selectbox(t("تعليم الآباء","Parent edu"), ["High School","Associate's","Bachelor's","Master's","PhD"])
            with q2: internet_in= st.selectbox(t("إنترنت","Internet"),          ["Yes","No"])
            with q3: extra_in   = st.selectbox(t("أنشطة","Extra"),              ["Yes","No"])
            card_close()

            predict_btn = st.button(f"⚡ {t('احسب التوقع لمادة','Run Prediction for')} {subj_in}",
                                    use_container_width=True, type="primary")

        with right:
            # Live score preview (always visible)
            tot_achieved = mid_in + fin_in + (asgn_in * 0.1) + (quiz_in * 0.1)
            tot_possible = 100
            live_est = (tot_achieved / tot_possible * 100)
            gl_live, gc_live = grade_letter(live_est)

            card_open()
            st.markdown(f"""
            <div style="text-align:center;padding:10px 0 6px;">
                <div style="font-size:.68rem;color:var(--text-3);text-transform:uppercase;letter-spacing:.12em;font-weight:700;margin-bottom:8px;">
                    {t("معاينة مباشرة","Live Preview")}
                </div>
                <div style="font-size:4rem;font-weight:900;color:{gc_live};letter-spacing:-.05em;line-height:1;">{gl_live}</div>
                <div style="font-size:.85rem;color:var(--text-3);margin-top:4px;">{live_est:.1f}/100</div>
            </div>
            <hr class="sc-hr">
            {score_bars_html({"Midterm_Score":mid_in,"Final_Score":fin_in,"Assignments_Avg":asgn_in,
                               "Quizzes_Avg":quiz_in,"Projects_Score":proj_in,"Participation_Score":part_in}, FEATURES)}
            """, unsafe_allow_html=True)
            card_close()

            if predict_btn:
                row = pd.DataFrame([{
                    'Gender':gender_in,'Age':age_in,'Department':dept_in,
                    'Attendance (%)':att_in,'Midterm_Score':mid_in,'Final_Score':fin_in,
                    'Assignments_Avg':asgn_in,'Quizzes_Avg':quiz_in,'Participation_Score':part_in,
                    'Projects_Score':proj_in,'Total_Score':live_est,
                    'Study_Hours_per_Week':hours_in,'Extracurricular_Activities':extra_in,
                    'Internet_Access_at_Home':internet_in,'Parent_Education_Level':parent_in,
                    'Family_Income_Level':income_in,'Stress_Level (1-10)':stress_in,
                    'Sleep_Hours_per_Night':sleep_in
                }])
                row['Score_Improvement']   = row['Final_Score'] - row['Midterm_Score']
                row['Assignment_Quiz_Avg'] = (row['Assignments_Avg'] + row['Quizzes_Avg'])/2
                row['Engagement_Score']    = (row['Participation_Score'] + row['Projects_Score'])/2
                row['Sleep_Study_Ratio']   = row['Sleep_Hours_per_Night']/(row['Study_Hours_per_Week']+1)
                row['High_Attendance']     = (row['Attendance (%)'] >= 75).astype(int)
                row['Low_Stress']          = (row['Stress_Level (1-10)'] <= 4).astype(int)
                enc = pd.get_dummies(row).reindex(columns=feature_names, fill_value=0)

                xp = xgb_model.predict(enc)[0];       xpr = xgb_model.predict_proba(enc)[0]
                _  = rf_model.predict(enc)[0];        rpr = rf_model.predict_proba(enc)[0]

                is_pass = xp == 1
                prob    = xpr[1]*100
                bg  = "rgba(18,201,138,0.1)"  if is_pass else "rgba(242,65,90,0.1)"
                bd  = "rgba(18,201,138,0.3)"  if is_pass else "rgba(242,65,90,0.3)"
                tc  = "#12c98a" if is_pass else "#f2415a"
                sym = "check_circle" if is_pass else "warning"
                lbl = t("ناجح","Pass") if is_pass else t("في خطر","At Risk")

                st.markdown(f"""
                <div class="result-wrap" style="background:{bg};border:1px solid {bd};">
                    <span class="ms" style="font-size:3rem!important;color:{tc};display:block;margin-bottom:8px;">{sym}</span>
                    <div class="status-tag" style="color:{tc};">{subj_in}</div>
                    <div class="big-num" style="color:{tc};">{lbl}</div>
                    <div style="font-size:.82rem;color:var(--text-3);margin:8px 0 16px;">{t("احتمالية النجاح","Pass Probability")}</div>
                    <div style="font-size:2.4rem;font-weight:900;color:{tc};margin-bottom:10px;">{prob:.1f}%</div>
                    <div style="background:rgba(255,255,255,0.05);border-radius:99px;height:8px;overflow:hidden;border:1px solid rgba(255,255,255,0.06);">
                        <div style="height:100%;width:{prob:.1f}%;background:linear-gradient(90deg,{tc}88,{tc});border-radius:99px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Model comparison
                card_open()
                section_label(t("مقارنة النماذج","Model Comparison"), "compare")
                for mn, pr, ak in [("XGBoost",xpr[1]*100,"xgb_acc"),("Random Forest",rpr[1]*100,"rf_acc")]:
                    acc = (metadata or {}).get(ak, 0)
                    st.markdown(f"""
                    <div style="margin-bottom:13px;">
                        <div style="display:flex;justify-content:space-between;font-size:.82rem;margin-bottom:6px;">
                            <span style="font-weight:700;color:var(--text-2);">{mn}</span>
                            <span style="color:var(--teal);font-weight:700;">{pr:.1f}% · {acc:.1%}</span>
                        </div>
                        <div class="sb-track"><div class="sb-fill" style="width:{pr:.1f}%;background:linear-gradient(90deg,#00dfc488,#4f8fff);"></div></div>
                    </div>""", unsafe_allow_html=True)
                card_close()

                # Top influencing factors
                card_open()
                section_label(t("أكثر العوامل تأثيراً","Top Influencing Factors"), "insights")
                fi  = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(7)
                fig, ax = dark_fig(5, 3)
                colors  = ['#00dfc4' if i==0 else '#4f8fff' if i<3 else '#444d78' for i in range(len(fi))]
                ax.barh(fi.index[::-1], fi.values[::-1], color=colors[::-1], height=0.55, edgecolor='none')
                ax.set_xlabel('Importance', fontsize=8); ax.set_xlim(0, fi.values.max()*1.2)
                for spine in ax.spines.values(): spine.set_visible(False)
                ax.tick_params(labelsize=8)
                plt.tight_layout(pad=0.5)
                st.pyplot(fig, use_container_width=True); plt.close()
                card_close()

            else:
                st.markdown(f"""
                <div class="tip-box" style="margin-top:0;">
                    <span class="ms">tips_and_updates</span>
                    <p>{t("اضبط السلايدرز من اليسار ثم اضغط «احسب التوقع» لترى نتيجتك الفورية.","Adjust the sliders on the left, then press «Run Prediction» to see your instant result.")}</p>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SCORE CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.nav_page == "calculator":
    st.markdown(f"""
    <div class="ph-wrap">
        <div class="ph-title">
            <div class="ph-icon" style="background:var(--amber-dim);border:1px solid rgba(244,162,45,0.22);">
                <span class="ms" style="color:var(--amber);font-size:24px!important">calculate</span>
            </div>
            <div>
                <h2 class="ph-h2">{t("حاسبة الدرجات الشاملة","Grade Calculator")}</h2>
                <div class="ph-sub">{t("احسب درجاتك واعرف كم تحتاج في الفاينل","Calculate your scores and know exactly what you need in the final")}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        card_open("amber")
        section_label(t("أدخل درجاتك","Enter Your Scores"), "edit")
        mid_c  = st.slider(t("الميدترم / 30", "Midterm / 30"),       0, 30,  min(int(st.session_state.manual_scores.get("Midterm_Score",0)), 30))
        asgn_c = st.slider(t("الواجبات / 100", "Assignments / 100"), 0, 100, min(int(st.session_state.manual_scores.get("Assignments_Avg",0)), 100))
        quiz_c = st.slider(t("الكويزات / 100", "Quizzes / 100"),     0, 100, min(int(st.session_state.manual_scores.get("Quizzes_Avg",0)), 100))
        divider()
        target = st.slider(t("الدرجة الكلية المستهدفة","Target Total Score"), 50, 100, 60)
        card_close()

        # Weight explanation
        card_open()
        section_label(t("توزيع الأوزان","Score Weights"), "pie_chart")
        weights = [("الميدترم","Midterm",30),("الفاينل","Final",50),("الواجبات","Assignments",10),("الكويزات","Quizzes",10)]
        fig, ax = dark_fig(5, 2.5)
        vals   = [30,50,10,10]
        colors = ['#00dfc4','#4f8fff','#b06ef5','#f4a22d']
        wedges, _ = ax.pie(vals, colors=colors, startangle=90,
                           wedgeprops=dict(width=0.55, edgecolor='#0b0e1c', linewidth=2))
        for w, lbl in zip(wedges, [t_plot(a,b) for a,b,_ in weights]):
            w.set_linewidth(0)
        ax.legend([t_plot(a,b) for a,b,_ in weights], loc="right", fontsize=8, frameon=False,
                  labelcolor='#8b92bc', bbox_to_anchor=(1.3, 0.5))
        ax.set_aspect('equal')
        plt.tight_layout(pad=0.2)
        st.pyplot(fig, use_container_width=True); plt.close()
        card_close()

    with c2:
        pts_mid  = mid_c
        pts_asgn = asgn_c * 0.1
        pts_quiz = quiz_c * 0.1
        pts_so_far = pts_mid + pts_asgn + pts_quiz
        
        needed_pts = target - pts_so_far
        needed_final = max(0.0, min(50.0, needed_pts))
        feasible  = needed_final <= 50
        confident = needed_final <= 35

        gc_r = "#12c98a" if confident else "#f4a22d" if feasible else "#f2415a"
        predicted_total = pts_so_far + needed_final
        gl_c, gc_c = grade_letter(predicted_total)

        card_open()
        st.markdown(f"""
        <div style="text-align:center;padding:8px 0 16px;">
            <span class="ms" style="font-size:2.5rem!important;color:{gc_r};display:block;margin-bottom:8px;">
                {'check_circle' if confident else 'warning' if feasible else 'cancel'}
            </span>
            <div style="font-size:.68rem;color:var(--text-3);text-transform:uppercase;letter-spacing:.12em;font-weight:700;margin-bottom:6px;">
                {t("تحتاج في الفاينل","Final Exam Needed")}
            </div>
            <div style="font-size:4rem;font-weight:900;color:{gc_r};line-height:1;">{needed_final:.0f}</div>
            <div style="font-size:.82rem;color:var(--text-3);margin-top:4px;">{t("من 50 درجة", "out of 50")}</div>
            <div style="margin-top:16px;">
                <span class="spill {'g' if confident else 'a' if feasible else 'r'}">
                    {t("ممكن جداً ✓","Very achievable ✓") if confident else t("ممكن ⚠","Possible ⚠") if feasible else t("صعب جداً ✗","Very difficult ✗")}
                </span>
            </div>
        </div>
        <hr class="sc-hr">
        """, unsafe_allow_html=True)

        # Breakdown table
        breakdown = [
            (t("ميدترم","Midterm"),      mid_c,  30,  pts_mid,   "#00dfc4"),
            (t("واجبات","Assignments"),  asgn_c, 100, pts_asgn,  "#b06ef5"),
            (t("كويزات","Quizzes"),      quiz_c, 100, pts_quiz,  "#f4a22d"),
        ]
        for lbl, raw, mx, pts, clr in breakdown:
            r = raw/mx if mx else 0
            st.markdown(f"""
            <div class="sb-wrap">
                <div class="sb-row">
                    <span class="sb-label" style="color:var(--text-2);">{lbl} ({raw}/{mx})</span>
                    <span class="sb-val" style="color:{clr};">+{pts:.1f}</span>
                </div>
                <div class="sb-track"><div class="sb-fill" style="width:{r*100:.0f}%;background:{clr};"></div></div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <hr class="sc-hr">
        <div style="display:flex;justify-content:space-between;font-size:.88rem;font-weight:700;">
            <span style="color:var(--text-2);">{t("المجموع بدون الفاينل","Total without final")}</span>
            <span style="color:var(--teal);">{pts_so_far:.1f}</span>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:.88rem;font-weight:700;margin-top:8px;">
            <span style="color:var(--text-2);">{t("التقدير المتوقع","Expected Grade")}</span>
            <span style="color:{gc_c};font-size:1.1rem;">{gl_c}</span>
        </div>
        """, unsafe_allow_html=True)
        card_close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.nav_page == "analytics":
    st.markdown(f"""
    <div class="ph-wrap">
        <div class="ph-title">
            <div class="ph-icon" style="background:var(--violet-dim);border:1px solid rgba(176,110,245,0.22);">
                <span class="ms" style="color:var(--violet);font-size:24px!important">bar_chart</span>
            </div>
            <div>
                <h2 class="ph-h2">{t("تحليل البيانات المتقدم","Advanced Analytics")}</h2>
                <div class="ph-sub">{t("إحصائيات شاملة وتصورات بيانية لأداء الطلاب","Comprehensive statistics and visualizations for student performance")}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df_dataset is None:
        st.error(t("ملف البيانات غير موجود في data/raw/","Dataset not found in data/raw/"))
    else:
        if 'Pass_Fail' not in df_dataset.columns:
            df_dataset['Pass_Fail'] = (df_dataset['Grade'] != 'F').astype(int)
        drop = ['Student_ID','First_Name','Last_Name','Email','Grade']
        dfc  = df_dataset.drop(columns=[c for c in drop if c in df_dataset.columns], errors='ignore')

        tabs = st.tabs([
            t("نظرة عامة","Overview"),
            t("التوزيعات","Distributions"),
            t("الارتباطات","Correlations"),
            t("تقييم النماذج","Model Eval"),
        ])

        # ── Tab 1: Overview ──────────────────────────────────────────────────
        with tabs[0]:
            n_total  = len(dfc)
            pass_r   = df_dataset['Pass_Fail'].mean()*100
            avg_sc   = dfc['Total_Score'].mean() if 'Total_Score' in dfc.columns else 0
            n_dept   = dfc['Department'].nunique() if 'Department' in dfc.columns else 0

            c1,c2,c3,c4 = st.columns(4)
            for col_w, icon, val, lbl, clr in [
                (c1,"people",        f"{n_total:,}",    t("إجمالي الطلاب","Students"), "#00dfc4"),
                (c2,"check_circle",  f"{pass_r:.1f}%",  t("نسبة النجاح","Pass Rate"),  "#12c98a"),
                (c3,"grade",         f"{avg_sc:.1f}",   t("متوسط الدرجات","Avg Score"), "#f4a22d"),
                (c4,"account_tree",  str(n_dept),       t("الأقسام","Departments"),     "#b06ef5"),
            ]:
                with col_w:
                    st.markdown(f"""
                    <div class="ana-stat">
                        <span class="ms ana-icon" style="color:{clr}">{icon}</span>
                        <div class="ana-val" style="color:{clr};">{val}</div>
                        <div class="ana-label">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            card_open()
            section_label(t("عينة من البيانات","Data Sample"), "table")
            st.dataframe(dfc.head(12), use_container_width=True, hide_index=True)
            card_close()

            card_open()
            r1, r2 = st.columns(2)
            with r1:
                section_label(t("القيم المفقودة","Missing Values"), "warning")
                st.dataframe(dfc.isnull().sum().reset_index().rename(columns={"index":"Column",0:"Missing"}),
                             use_container_width=True, hide_index=True)
            with r2:
                section_label(t("إحصائيات وصفية","Descriptive Stats"), "analytics")
                st.dataframe(dfc.describe().round(2), use_container_width=True)
            card_close()

        # ── Tab 2: Distributions ─────────────────────────────────────────────
        with tabs[1]:
            r1c1, r1c2 = st.columns(2)

            with r1c1:
                card_open()
                section_label(t("توزيع درجة الإجمالي","Total Score Distribution"), "bar_chart")
                if 'Total_Score' in dfc.columns:
                    fig, ax = dark_fig(5, 3.5)
                    ax.hist(dfc['Total_Score'].dropna(), bins=28, color='#00dfc4', alpha=0.7, edgecolor='none')
                    # KDE overlay
                    from scipy.stats import gaussian_kde
                    kde_x = np.linspace(dfc['Total_Score'].min(), dfc['Total_Score'].max(), 200)
                    kde   = gaussian_kde(dfc['Total_Score'].dropna())
                    ax2   = ax.twinx()
                    ax2.plot(kde_x, kde(kde_x), color='#4f8fff', lw=2)
                    ax2.set_yticks([]); ax2.set_facecolor('#0b0e1c')
                    ax.set_xlabel(t_plot("الدرجة الكلية","Total Score")); ax.set_ylabel(t_plot("العدد","Count"))
                    for sp in ax.spines.values(): sp.set_visible(False)
                    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
                card_close()

            with r1c2:
                card_open()
                section_label(t("نسبة النجاح حسب القسم","Pass Rate by Department"), "school")
                if 'Department' in dfc.columns:
                    dept_pr = df_dataset.groupby('Department')['Pass_Fail'].mean().sort_values()*100
                    fig, ax = dark_fig(5, 3.5)
                    clrs = ['#12c98a' if v>=70 else '#f4a22d' if v>=50 else '#f2415a' for v in dept_pr.values]
                    bars = ax.barh(dept_pr.index, dept_pr.values, color=clrs, height=0.55, edgecolor='none')
                    ax.axvline(60, color='#444d78', ls='--', lw=1, alpha=0.7)
                    for i,(v,b) in enumerate(zip(dept_pr.values,bars)):
                        ax.text(v+0.5, b.get_y()+b.get_height()/2, f'{v:.0f}%',
                                va='center', fontsize=8, color='#8b92bc')
                    ax.set_xlabel(t_plot("نسبة النجاح %","Pass Rate %"))
                    for sp in ax.spines.values(): sp.set_visible(False)
                    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
                card_close()

            card_open()
            section_label(t("توزيع الدرجات حسب القسم والتقدير","Grade Distribution by Department"), "table_chart")
            if 'Department' in df_dataset.columns and 'Grade' in df_dataset.columns:
                fig, ax = dark_fig(9, 4)
                dept_grade = df_dataset.groupby(['Department','Grade']).size().unstack(fill_value=0)
                grade_colors = {'A':'#12c98a','B':'#00dfc4','C':'#4f8fff','D':'#f4a22d','F':'#f2415a'}
                bottom = np.zeros(len(dept_grade))
                for grade in dept_grade.columns:
                    clr = grade_colors.get(grade, '#8b92bc')
                    ax.bar(dept_grade.index, dept_grade[grade], bottom=bottom,
                           color=clr, label=grade, alpha=0.85, edgecolor='none')
                    bottom += dept_grade[grade].values
                ax.set_xlabel(t_plot("القسم","Department")); ax.set_ylabel(t_plot("عدد الطلاب","Students"))
                ax.legend(loc='upper right', frameon=False, labelcolor='#8b92bc', fontsize=9)
                ax.tick_params(axis='x', rotation=30)
                for sp in ax.spines.values(): sp.set_visible(False)
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
            card_close()

        # ── Tab 3: Correlations ──────────────────────────────────────────────
        with tabs[2]:
            card_open()
            section_label(t("مصفوفة الارتباط","Correlation Matrix"), "grid_on")
            num_cols = dfc.select_dtypes(include=np.number).columns.tolist()[:10]
            if num_cols:
                corr = dfc[num_cols].corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                fig, ax = dark_fig(9, 5)
                cmap = sns.diverging_palette(155, 260, s=90, l=40, as_cmap=True)
                sns.heatmap(corr, mask=mask, cmap=cmap, center=0, ax=ax,
                            linewidths=0.4, linecolor='#111526',
                            cbar_kws={"shrink":0.7,"pad":0.02},
                            annot=True, fmt='.2f', annot_kws={"size":8,"color":"#edf0ff"})
                ax.tick_params(axis='x', rotation=30, labelsize=8)
                ax.tick_params(axis='y', rotation=0, labelsize=8)
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(colors='#8b92bc', labelsize=8)
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
            card_close()

            if 'Total_Score' in dfc.columns and 'Study_Hours_per_Week' in dfc.columns:
                card_open()
                r_sc, r_sl = st.columns(2)
                with r_sc:
                    section_label(t("الدراسة vs الدرجات","Study Hours vs Score"), "trending_up")
                    fig, ax = dark_fig(5, 3.5)
                    ax.scatter(dfc['Study_Hours_per_Week'], dfc['Total_Score'],
                               alpha=0.4, s=18, c='#00dfc4', edgecolors='none')
                    m, b = np.polyfit(dfc['Study_Hours_per_Week'].dropna(), dfc['Total_Score'].dropna(), 1)
                    xr = np.linspace(dfc['Study_Hours_per_Week'].min(), dfc['Study_Hours_per_Week'].max(), 100)
                    ax.plot(xr, m*xr+b, color='#4f8fff', lw=2)
                    ax.set_xlabel(t_plot("ساعات الدراسة","Study Hours"))
                    ax.set_ylabel(t_plot("الدرجة الكلية","Total Score"))
                    for sp in ax.spines.values(): sp.set_visible(False)
                    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

                with r_sl:
                    if 'Stress_Level (1-10)' in dfc.columns:
                        section_label(t("التوتر vs الدرجات","Stress vs Score"), "mood_bad")
                        fig, ax = dark_fig(5, 3.5)
                        ax.scatter(dfc['Stress_Level (1-10)'], dfc['Total_Score'],
                                   alpha=0.4, s=18, c='#f2415a', edgecolors='none')
                        ax.set_xlabel(t_plot("مستوى التوتر","Stress Level"))
                        ax.set_ylabel(t_plot("الدرجة الكلية","Total Score"))
                        for sp in ax.spines.values(): sp.set_visible(False)
                        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
                card_close()

        # ── Tab 4: Model Evaluation ──────────────────────────────────────────
        with tabs[3]:
            if metadata:
                m1,m2,m3,m4 = st.columns(4)
                metric_items = [
                    (m1,"XGBoost Acc",  f"{metadata.get('xgb_acc',0):.2%}", "#00dfc4"),
                    (m2,"RF Accuracy",  f"{metadata.get('rf_acc',0):.2%}",  "#4f8fff"),
                    (m3,"XGB ROC-AUC",  f"{metadata.get('xgb_auc',0):.4f}", "#b06ef5"),
                    (m4,"RF ROC-AUC",   f"{metadata.get('rf_auc',0):.4f}",  "#f4a22d"),
                ]
                for col_w, lbl, val, clr in metric_items:
                    with col_w:
                        st.markdown(f"""
                        <div class="ana-stat" style="border-top:2px solid {clr};">
                            <div class="ana-val" style="color:{clr};">{val}</div>
                            <div class="ana-label">{lbl}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                if xgb_model and rf_model and feature_names:
                    fe1, fe2 = st.columns(2)
                    for cw, mod, mn, clr in [(fe1,xgb_model,"XGBoost","#00dfc4"),(fe2,rf_model,"Random Forest","#4f8fff")]:
                        with cw:
                            card_open()
                            section_label(f"{mn} — {t('أهمية المتغيرات','Feature Importance')}", "insights")
                            fi = pd.Series(mod.feature_importances_, index=feature_names).sort_values(ascending=False).head(9)
                            fig, ax = dark_fig(5, 4)
                            bar_c = [clr if i==0 else f"{clr}99" if i<4 else '#444d78' for i in range(len(fi))]
                            ax.barh(fi.index[::-1], fi.values[::-1], color=bar_c[::-1], height=0.55, edgecolor='none')
                            for sp in ax.spines.values(): sp.set_visible(False)
                            ax.set_xlabel('Importance', fontsize=8); ax.tick_params(labelsize=8)
                            plt.tight_layout(pad=0.5); st.pyplot(fig, use_container_width=True); plt.close()
                            card_close()
            else:
                st.info(t("دُرّب النماذج أولاً لرؤية نتائج التقييم","Train models first to see evaluation results."))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STUDY TIPS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.nav_page == "tips":
    st.markdown(f"""
    <div class="ph-wrap">
        <div class="ph-title">
            <div class="ph-icon" style="background:rgba(244,162,45,0.1);border:1px solid rgba(244,162,45,0.22);">
                <span class="ms" style="color:var(--amber);font-size:24px!important">lightbulb</span>
            </div>
            <div>
                <h2 class="ph-h2">{t("نصائح الدراسة الذكية","Smart Study Tips")}</h2>
                <div class="ph-sub">{t("استراتيجيات مبنية على أبحاث علم الأعصاب وبيانات الأداء الأكاديمي","Strategies based on neuroscience research and academic performance data")}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tips_data = [
        ("schedule",       "#00dfc4",
         t("الاسترجاع النشط","Active Recall"),
         t("اختبر نفسك بدلاً من إعادة القراءة. البطاقات التعليمية والاختبارات الذاتية تزيد الاحتفاظ بالمعلومات بنسبة تصل لـ 50%.","Test yourself instead of re-reading. Flashcards and self-quizzing boost retention by up to 50% compared to passive review.")),
        ("timer",          "#4f8fff",
         t("تقنية بومودورو","Pomodoro Technique"),
         t("اشتغل 25 دقيقة، استرح 5 دقائق. بعد 4 دورات خذ راحة 30 دقيقة. يحافظ على التركيز ويمنع الإنهاك الذهني.","Work 25 min, rest 5 min. After 4 cycles, take a 30-min break. Maintains focus and prevents cognitive burnout.")),
        ("bedtime",        "#b06ef5",
         t("النوم والتعزيز","Sleep & Memory"),
         t("7-9 ساعات نوم ليست رفاهية. تعزيز الذاكرة يحدث أثناء النوم العميق — المذاكرة الساعة 3 صباحاً تعطي نتيجة عكسية.","7-9 hours of sleep is non-negotiable. Memory consolidation happens during deep sleep — cramming at 3am is counterproductive.")),
        ("fitness_center", "#f2415a",
         t("الرياضة والدماغ","Exercise & Brain"),
         t("30 دقيقة رياضة معتدلة 3 مرات أسبوعياً ترفع BDNF (هرمون نمو الدماغ) بنسبة 20%. يحسّن التركيز والذاكرة مباشرة.","30 min moderate exercise 3x/week raises BDNF by 20%, directly improving focus and memory formation.")),
        ("groups",         "#12c98a",
         t("مجموعات الدراسة","Study Groups"),
         t("شرح المفاهيم للآخرين يعزز فهمك بنسبة 90% (مقارنة بـ 10% من القراءة فقط). كوّن مجموعات من 3-4 أشخاص بأهداف واضحة.","Teaching others locks in 90% retention vs 10% from reading alone. Form groups of 3-4 with focused goals.")),
        ("psychology",     "#f4a22d",
         t("التوجه نحو النمو","Growth Mindset"),
         t("الإيمان بأنك تستطيع التحسن يتنبأ بالنجاح أكثر من معدل الذكاء. أعد صياغة «فشلت» كـ«لم أتقنها بعد» وستتغير النتائج.","Believing you can improve predicts success better than IQ. Reframe 'I failed' as 'I haven't mastered this yet.'")),
        ("water_drop",     "#00dfc4",
         t("الترطيب والتركيز","Hydration"),
         t("انخفاض 2% في مستوى الماء بالجسم يقلل التركيز بنسبة 20%. اشرب 8 أكواب يومياً خاصة وقت المذاكرة.","Even 2% dehydration reduces focus by 20%. Drink 8 glasses daily, especially during study sessions.")),
        ("edit_note",      "#b06ef5",
         t("التعلم التباعدي","Spaced Repetition"),
         t("راجع المادة بعد يوم، ثم أسبوع، ثم شهر. هذا الأسلوب يحسّن الاحتفاظ بالمعلومات على المدى البعيد بشكل كبير.","Review material after 1 day, 1 week, 1 month. Spaced repetition dramatically improves long-term retention.")),
    ]

    c1, c2 = st.columns(2)
    for i, (icon, color, title, body) in enumerate(tips_data):
        with (c1 if i%2==0 else c2):
            st.markdown(f"""
            <div class="tip-card">
                <div class="tc-icon" style="background:{color}15;border:1px solid {color}30;">
                    <span class="ms" style="color:{color};font-size:22px!important">{icon}</span>
                </div>
                <div class="tc-title">{title}</div>
                <div class="tc-body">{body}</div>
            </div>""", unsafe_allow_html=True)

    # Personalized tip based on scores
    ms_ = st.session_state.manual_scores
    personal_tips = []
    
    mid_pct = (ms_["Midterm_Score"] / 30) * 100
    if ms_["Midterm_Score"] > 0 and mid_pct < 60:
        personal_tips.append(t(f"ميدترمك ({ms_['Midterm_Score']}/30) يحتاج تحسين — ركز على الأسئلة القصيرة والمفاهيم الأساسية",
                               f"Your midterm ({ms_['Midterm_Score']}/30) needs work — focus on short questions and core concepts"))
                               
    asgn_pct = (ms_["Assignments_Avg"] / 100) * 100
    if ms_["Assignments_Avg"] > 0 and asgn_pct < 60:
        personal_tips.append(t(f"أداء الواجبات ({ms_['Assignments_Avg']}/100) — حاول تخصيص وقت ثابت لحل الواجبات لتفادي التراكم",
                               f"Assignments performance ({ms_['Assignments_Avg']}/100) — allocate dedicated time to avoid pile-ups"))
                               
    quiz_pct = (ms_["Quizzes_Avg"] / 100) * 100
    if ms_["Quizzes_Avg"] > 0 and quiz_pct < 60:
        personal_tips.append(t(f"الكويزات ({ms_['Quizzes_Avg']}/100) — جرّب تقنية الاسترجاع النشط قبل كل كويز",
                               f"Quizzes ({ms_['Quizzes_Avg']}/100) — try the active recall technique before each quiz"))

    if personal_tips:
        st.markdown("<br>", unsafe_allow_html=True)
        card_open("teal")
        section_label(t("توصيات شخصية بناءً على درجاتك","Personalized Recommendations"), "person")
        for tip in personal_tips:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:10px;">
                <span class="ms" style="color:var(--teal);font-size:16px!important;margin-top:1px;">arrow_forward</span>
                <span style="font-size:.84rem;color:var(--text-2);line-height:1.55;">{tip}</span>
            </div>""", unsafe_allow_html=True)
        card_close()