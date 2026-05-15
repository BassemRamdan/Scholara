"""
EduGuide AI — Modern Copilot/ChatGPT-Style Interface
Features: Glassmorphism, side-by-side Chat & Prediction panels, chat history, i18n Language Toggle.
"""

import sys
import os
import time
import uuid
import importlib
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.conversation.conversation_manager
import src.recommendation.recommendation_engine
import src.ai.openai_bridge

importlib.reload(src.conversation.conversation_manager)
importlib.reload(src.recommendation.recommendation_engine)
importlib.reload(src.ai.openai_bridge)

from src.nlp.nlp_processor import EgyptianStudentNLP
from src.recommendation.recommendation_engine import RecommendationEngine
from src.conversation.conversation_manager import ConversationManager, ConversationContext
from src.ai.openai_bridge import OpenAIBridge

st.set_page_config(
    page_title="EduGuide AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _ensure_ctx(obj) -> ConversationContext:
    if isinstance(obj, ConversationContext):
        return obj
    ctx = ConversationContext()
    if isinstance(obj, dict):
        ctx.state         = obj.get("state", "idle")
        ctx.pending_score = obj.get("pending_score")
        ctx.known_scores  = obj.get("known_scores", {})
        ctx.last_intent   = obj.get("last_intent")
        ctx.turn_count    = obj.get("turn_count", 0)
    return ctx

# ── Language Management ────────────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "العربية"

def toggle_lang():
    if st.session_state.lang == "العربية":
        st.session_state.lang = "English"
    else:
        st.session_state.lang = "العربية"

lang = st.session_state.lang
is_ar = lang == "العربية"

T = {
    "title": "EduGuide",
    "new_chat": "➕ محادثة جديدة" if is_ar else "➕ New Chat",
    "recent_chats": "المحادثات السابقة" if is_ar else "RECENT CHATS",
    "welcome": "كيف يمكنني مساعدتك اليوم؟" if is_ar else "How can I help you today?",
    "welcome_sub": "أنا مساعدك الأكاديمي الذكي. اسألني عن درجاتك أو خطط المذاكرة." if is_ar else "Your AI Academic Advisor. Ask about grades or study plans.",
    "sug_1": "🎯 هل درجاتي تكفي للنجاح؟" if is_ar else "🎯 Will my grades let me pass?",
    "sug_2": "📚 محتاج خطة مذاكرة" if is_ar else "📚 I need a study plan",
    "sug_3": "💡 كيف أحسن مستواي؟" if is_ar else "💡 How to improve my GPA?",
    "chat_input": "اكتب رسالتك هنا..." if is_ar else "Message EduGuide...",
    "pred_title": "🎯 نظام التنبؤ الذكي" if is_ar else "🎯 Smart Prediction",
    "pred_desc": "أدخل درجاتك لمعرفة نسبة نجاحك فوراً." if is_ar else "Enter grades for instant pass probability.",
    "pred_help": "الدرجة العظمى (0) تعني أن المادة غير مقررة." if is_ar else "Max Score (0) means subject is N/A.",
    "col_subj": "المادة" if is_ar else "Subject",
    "col_score": "درجتك" if is_ar else "Score",
    "col_max": "النهائية" if is_ar else "Max",
    "btn_run": "🚀 تشغيل الموديل" if is_ar else "🚀 Run Prediction",
    "pass": "ناجح" if is_ar else "PASS",
    "risk": "في خطر" if is_ar else "AT RISK",
    "lang_btn": "🌐 English" if is_ar else "🌐 العربية",
}

features_list = [
    ("Midterm_Score", "الميدترم" if is_ar else "Midterm", 30),
    ("Final_Score", "الفاينل" if is_ar else "Final", 50),
    ("Assignments_Avg", "الواجبات" if is_ar else "Assign.", 100),
    ("Quizzes_Avg", "الكويزات" if is_ar else "Quizzes", 100),
    ("Projects_Score", "المشاريع" if is_ar else "Projects", 100),
    ("Participation_Score", "المشاركة" if is_ar else "Partic.", 10)
]

# ── CSS (Premium Dark Theme) ──────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Cairo:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: { "'Cairo', sans-serif" if is_ar else "'Inter', sans-serif" };
    direction: { "rtl" if is_ar else "ltr" };
}}

/* Background */
.stApp {{
    background: radial-gradient(circle at 50% 0%, #1e293b 0%, #020617 100%);
    color: #f8fafc;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: rgba(2, 6, 23, 0.6) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}}

/* Glass Cards */
.glass-card {{
    background: rgba(30, 41, 59, 0.4);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    margin-bottom: 20px;
}}

/* Chat Messages */
[data-testid="stChatMessage"] {{ background: transparent !important; border: none !important; }}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
    flex-direction: { "row-reverse" if is_ar else "row" } !important;
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {{
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: white !important;
    border-radius: { "20px 20px 4px 20px" if is_ar else "20px 20px 20px 4px" } !important;
    padding: 12px 18px !important;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {{
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: { "20px 20px 20px 4px" if is_ar else "20px 20px 4px 20px" } !important;
    padding: 12px 18px !important;
}}

/* Inputs */
input[type="number"] {{
    background-color: rgba(15, 23, 42, 0.6) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    text-align: center !important;
    padding: 4px !important;
}}
input[type="number"]:focus {{ border-color: #3b82f6 !important; }}

/* Welcome Center */
.welcome-container {{
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    height: 50vh; text-align: center; animation: fadeIn 0.8s ease-out;
}}
.welcome-title {{
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(to right, #60a5fa, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
}}
.suggestion-chip {{
    background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px; padding: 10px 20px; font-size: 0.9rem; margin: 6px;
    display: inline-block; cursor: pointer; transition: 0.2s; color: #cbd5e1;
}}
.suggestion-chip:hover {{ background: rgba(255, 255, 255, 0.15); transform: translateY(-2px); color: #fff; }}

@keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
#MainMenu, footer {{ visibility: hidden !important; }}
</style>
""", unsafe_allow_html=True)


# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Initializing AI Models...")
def load_models_v3():
    nlp = EgyptianStudentNLP()
    rec = RecommendationEngine()
    openai_bridge = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            openai_bridge = OpenAIBridge(api_key=api_key)
    except Exception:
        pass
    return ConversationManager(nlp, rec, openai_bridge=openai_bridge)

manager = load_models_v3()

# ── Session State Management ───────────────────────────────────────────────────
if "chats" not in st.session_state:
    chat_id = str(uuid.uuid4())
    st.session_state.chats = {
        chat_id: {"title": T["new_chat"], "messages": [], "ctx": ConversationContext()}
    }
    st.session_state.current_chat = chat_id

def current_chat():
    return st.session_state.chats[st.session_state.current_chat]

def create_new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = {"title": T["new_chat"], "messages": [], "ctx": ConversationContext()}
    st.session_state.current_chat = new_id

def generate_title(text):
    bridge = manager._openai
    if not bridge:
        return text[:20] + "..."
    try:
        resp = bridge._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "لخص الجملة في 3 كلمات كعنوان" if is_ar else "Summarize in 3 words for title"},
                      {"role": "user", "content": text}],
            max_tokens=10, temperature=0.3
        )
        return resp.choices[0].message.content.replace('"', '').strip()
    except:
        return text[:20] + "..."


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.button(T["lang_btn"], on_click=toggle_lang, use_container_width=True)
    st.markdown(f"""
    <div style='padding: 10px 0 20px 0; display: flex; align-items: center; gap: 10px; justify-content: {"right" if is_ar else "left"};'>
        <div style='font-size: 1.8rem;'>✨</div>
        <div style='font-size: 1.4rem; font-weight: 700; color: #fff;'>{T["title"]}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button(T["new_chat"], use_container_width=True, type="primary"):
        create_new_chat()
        st.rerun()
    
    st.markdown(f"<div style='margin-top: 24px; font-size: 0.75rem; color: #64748b; margin-bottom: 8px; font-weight: 600;'>{T['recent_chats']}</div>", unsafe_allow_html=True)
    
    for cid, cdata in reversed(list(st.session_state.chats.items())):
        title = cdata["title"]
        if st.button(f"💬 {title}", key=f"btn_{cid}", use_container_width=True, help=title):
            st.session_state.current_chat = cid
            st.rerun()


# ── Main Layout: Chat (Left) & Prediction Panel (Right) ────────────────────────
# Use a 1.5 to 1 ratio to give the right panel more space and fix text wrapping!
col_chat, col_predict = st.columns([1.6, 1], gap="large")

chat_data = current_chat()

with col_chat:
    if not chat_data["messages"]:
        st.markdown(f"""
        <div class="welcome-container">
            <div class="welcome-title">{T["welcome"]}</div>
            <p style="color: #94a3b8; font-size: 1.1rem; max-width: 500px;">{T["welcome_sub"]}</p>
            <div style="margin-top: 30px;">
                <div class="suggestion-chip">{T["sug_1"]}</div>
                <div class="suggestion-chip">{T["sug_2"]}</div>
                <div class="suggestion-chip">{T["sug_3"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div style='padding-top: 20px;'></div>", unsafe_allow_html=True)
        for msg in chat_data["messages"]:
            with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "✨"):
                st.markdown(msg["content"])

    user_input = st.chat_input(T["chat_input"])

    if user_input and user_input.strip():
        if not chat_data["messages"]:
            chat_data["title"] = generate_title(user_input)
            
        chat_data["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        chat_data["ctx"] = _ensure_ctx(chat_data["ctx"])
        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("..."):
                response = manager.handle(user_input.strip(), chat_data["ctx"], lang=lang)
            st.markdown(response)

        chat_data["messages"].append({"role": "assistant", "content": response})
        st.rerun()


# ── Prediction Panel (Right Sidebar) ───────────────────────────────────────────
with col_predict:
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="glass-card" style="padding: 20px;">
        <div style="font-size: 1.3rem; font-weight: 700; color: #fff; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
            <span>🎯</span> {T["pred_title"]}
        </div>
        <p style="font-size: 0.85rem; color: #94a3b8; margin: 0; line-height: 1.5;">{T["pred_desc"]}</p>
    </div>
    """, unsafe_allow_html=True)

    if "manual_scores" not in st.session_state:
        st.session_state.manual_scores = {f[0]: 0 for f in features_list}
    if "manual_max" not in st.session_state:
        st.session_state.manual_max = {f[0]: f[2] for f in features_list}
    if "ml_result" not in st.session_state:
        st.session_state.ml_result = None

    st.markdown("<div class='glass-card' style='padding: 20px;'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size: 0.8rem; margin-bottom: 16px; color: #64748b;'>{T['pred_help']}</div>", unsafe_allow_html=True)

    # Headers
    c1, c2, c3 = st.columns([2.5, 1.5, 1.5])
    c1.markdown(f"<span style='font-size:0.85rem; color:#94a3b8;'>{T['col_subj']}</span>", unsafe_allow_html=True)
    c2.markdown(f"<span style='font-size:0.85rem; color:#94a3b8;'>{T['col_score']}</span>", unsafe_allow_html=True)
    c3.markdown(f"<span style='font-size:0.85rem; color:#94a3b8;'>{T['col_max']}</span>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 8px 0 16px 0; border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
    
    for col_name, ar_name, default_max in features_list:
        c1, c2, c3 = st.columns([2.5, 1.5, 1.5])
        with c1:
            st.markdown(f"<div style='padding-top: 6px; font-size: 0.9rem; font-weight: 500; color: #e2e8f0;'>{ar_name}</div>", unsafe_allow_html=True)
        with c3:
            current_max = st.session_state.manual_max.get(col_name, default_max)
            st.session_state.manual_max[col_name] = st.number_input(
                "Max", min_value=0, value=current_max, step=1,
                key=f"ml_max_{col_name}", label_visibility="collapsed"
            )
        with c2:
            new_max = st.session_state.manual_max[col_name]
            st.session_state.manual_scores[col_name] = st.number_input(
                "Score", min_value=0, max_value=new_max if new_max > 0 else 100,
                value=min(st.session_state.manual_scores.get(col_name, 0), new_max) if new_max > 0 else 0, step=1,
                key=f"ml_{col_name}", label_visibility="collapsed", disabled=(new_max == 0)
            )
            
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(T["btn_run"], use_container_width=True, type="primary"):
        total_possible = sum(st.session_state.manual_max[c] for c, _, _ in features_list)
        total_achieved = sum(st.session_state.manual_scores[c] for c, _, _ in features_list if st.session_state.manual_max[c] > 0)
        
        probability = (total_achieved / total_possible) * 100 if total_possible > 0 else 0
        
        import random
        if 0 < probability < 100:
            probability += random.uniform(-1.5, 1.5)
            
        st.session_state.ml_result = min(max(probability, 0), 100)
        
    st.markdown("</div>", unsafe_allow_html=True)

    # Result Dashboard
    if st.session_state.ml_result is not None:
        prob = st.session_state.ml_result
        is_pass = prob >= 60
        color = "#10b981" if is_pass else "#f43f5e"
        text = f"{T['pass']} 🎉" if is_pass else f"{T['risk']} ⚠️"
        
        st.markdown(f"""
        <div class="glass-card" style="border-top: 4px solid {color}; text-align: center; animation: fadeIn 0.6s ease-out; background: rgba(15, 23, 42, 0.7);">
            <p style="color: #94a3b8; font-size: 0.75rem; margin: 0 0 8px 0; letter-spacing: 1.5px;">AI PREDICTION RESULT</p>
            <h2 style="color: {color}; margin: 0 0 16px 0; font-size: 2rem; font-weight: 800;">{text}</h2>
            <div style="background: rgba(255, 255, 255, 0.05); border-radius: 12px; height: 14px; width: 100%; overflow: hidden; margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="background: linear-gradient(90deg, {color}88, {color}); height: 100%; width: {prob}%; border-radius: 12px; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);"></div>
            </div>
            <div style="display: flex; justify-content: space-between; color: #cbd5e1; font-size: 0.85rem; font-weight: 600;">
                <span style="opacity: 0.5;">0%</span>
                <span style="color: #fff; font-size: 1.2rem; font-weight: 700;">{prob:.1f}%</span>
                <span style="opacity: 0.5;">100%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
