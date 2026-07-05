"""
EmoMemory ✦ Premium Streamlit Application
Memory-Enabled Emotion Intelligence powered by Cognee Cloud
Redesigned with glassmorphism dark theme & Fluent/Noto emoji pack
"""

import streamlit as st
import os
from datetime import datetime
import logging

# Load .env for local development (optional — Streamlit Cloud uses st.secrets)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not required on Streamlit Cloud

# Configure Cognee environment before importing it
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"

# Use absolute writable temp directory paths for Cognee storage to avoid Permission Denied on Streamlit Cloud
import tempfile
temp_dir = tempfile.gettempdir()
os.environ["SYSTEM_ROOT_DIRECTORY"] = os.path.abspath(os.path.join(temp_dir, ".cognee_system"))
os.environ["system_root_directory"] = os.path.abspath(os.path.join(temp_dir, ".cognee_system"))
os.environ["DATA_ROOT_DIRECTORY"]   = os.path.abspath(os.path.join(temp_dir, ".cognee_data"))
os.environ["data_root_directory"]   = os.path.abspath(os.path.join(temp_dir, ".cognee_data"))
os.environ["COGNEE_LOGS_DIR"]       = os.path.abspath(os.path.join(temp_dir, ".cognee_logs"))

# ── Load secrets → environment (server-side only, never shown in UI) ──────────
# Streamlit secrets are injected at deploy time via the Streamlit Cloud dashboard.
# They are NEVER exposed to the browser or frontend.
try:
    for _k, _v in st.secrets.items():
        os.environ[_k] = str(_v)
except Exception:
    pass

# ── Configure Cognee to use Gemini (free) instead of OpenAI ──────────────────
# Cognee supports multiple LLM backends. We prefer Gemini because it has a
# generous free tier and doesn’t require any paid subscription.
_gemini_key = os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))
if _gemini_key:
    os.environ["LLM_PROVIDER"]  = "google"
    os.environ["LLM_MODEL"]     = os.getenv("LLM_MODEL", "gemini-2.0-flash-lite")
    os.environ["LLM_API_KEY"]   = _gemini_key
    os.environ["GOOGLE_API_KEY"]= _gemini_key
else:
    # No LLM key — set a dummy provider so Cognee starts without crashing
    os.environ.setdefault("LLM_PROVIDER", "openai")   # will be bypassed below
    os.environ.setdefault("LLM_MODEL",    "gpt-4o-mini")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Unique Emoji Pack ── (Noto / Fluent Unicode blocks, NOT standard emoji)
EMO_ICONS = {
    "happy":     "𝌆",   # tai xuan jing symbol (joy)
    "joy":       "𝌀",
    "sad":       "𝍢",   # domino tile
    "angry":     "𝍩",
    "fear":      "𝍫",
    "surprise":  "𝌌",
    "neutral":   "𝌎",
    "disgust":   "𝌑",
    "excited":   "𝌔",
    "calm":      "𝌕",
    "default":   "𝌈",
}

# Map plain labels → display emoji
EMOTION_EMOJI_MAP = {
    "happy":    ("𝌆",  "#f6e05e"),
    "joy":      ("𝌀",  "#f6e05e"),
    "sad":      ("𝍢",  "#90cdf4"),
    "sadness":  ("𝍢",  "#90cdf4"),
    "angry":    ("𝍩",  "#fc8181"),
    "anger":    ("𝍩",  "#fc8181"),
    "fear":     ("𝍫",  "#b794f4"),
    "surprise": ("𝌌",  "#f6ad55"),
    "neutral":  ("𝌎",  "#a0aec0"),
    "disgust":  ("𝌑",  "#68d391"),
    "excited":  ("𝌔",  "#fbb6ce"),
    "calm":     ("𝌕",  "#76e4f7"),
}

def get_emotion_style(emotion: str):
    key = emotion.lower()
    icon, color = EMOTION_EMOJI_MAP.get(key, ("𝌈", "#667eea"))
    return icon, color

# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="EmoMemory ✦ AI That Never Forgets",
    page_icon="𝌆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────
#  PREMIUM DARK GLASSMORPHISM CSS
# ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

/* ── Root variables ── */
:root {
    --bg-deep:     #060612;
    --bg-mid:      #0d0d1f;
    --bg-card:     rgba(13,13,31,0.72);
    --accent-1:    #7c5cfc;
    --accent-2:    #e040fb;
    --accent-3:    #00e5ff;
    --glow-1:      rgba(124,92,252,0.35);
    --glow-2:      rgba(224,64,251,0.2);
    --text-hi:     #f0f0ff;
    --text-mid:    #b0b0cc;
    --text-lo:     #606080;
    --border:      rgba(124,92,252,0.22);
    --radius:      18px;
    --radius-sm:   10px;
}

/* ── Global reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: var(--bg-deep) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-hi) !important;
}

/* Animated mesh background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(124,92,252,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%, rgba(224,64,251,0.12) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 50% 50%, rgba(0,229,255,0.06) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #08081a 0%, #0d0d22 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-hi) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ═══════════════════════════════════════════════════════
   BULLETPROOF FIX: Sidebar collapse arrow "keyboard_double_arrow_left"
   We completely hide all child elements/text of collapse buttons,
   then render a custom Unicode arrow in the pseudo-element.
   ═══════════════════════════════════════════════════════ */

/* 1. Hide all inner contents (text / icons) of Streamlit collapse/expand buttons */
[data-testid="stSidebarCollapseButton"] button *,
[data-testid="stSidebarCollapsedControl"] button *,
[data-testid="collapsedControl"] button *,
button[aria-label*="sidebar"] *,
button[aria-label*="Sidebar"] *,
button[aria-label*="Close"] *,
button[aria-label*="Open"] * {
    display: none !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
}

/* 2. Style the buttons and inject custom Unicode arrow */
[data-testid="stSidebarCollapseButton"] button,
button[aria-label*="Close"] {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: rgba(124,92,252,0.15) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    width: 36px !important;
    height: 36px !important;
    cursor: pointer !important;
    position: relative !important;
}
[data-testid="stSidebarCollapseButton"] button::before,
button[aria-label*="Close"]::before {
    content: "❮" !important;
    font-size: 15px !important;
    font-family: 'Inter', Arial, sans-serif !important;
    color: var(--text-hi) !important;
    display: block !important;
}

[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button,
button[aria-label*="Open"] {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: rgba(124,92,252,0.15) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    width: 36px !important;
    height: 36px !important;
    cursor: pointer !important;
    position: relative !important;
}
[data-testid="stSidebarCollapsedControl"] button::before,
[data-testid="collapsedControl"] button::before,
button[aria-label*="Open"]::before {
    content: "❯" !important;
    font-size: 15px !important;
    font-family: 'Inter', Arial, sans-serif !important;
    color: var(--text-hi) !important;
    display: block !important;
}

/* Hover effects for collapse buttons */
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="collapsedControl"] button:hover,
button[aria-label*="Close"]:hover,
button[aria-label*="Open"]:hover {
    background: rgba(124,92,252,0.35) !important;
    border-color: var(--accent-1) !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    background: rgba(20,20,44,0.85) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-hi) !important;
}
[data-testid="stSidebar"] button {
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2)) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    color: #fff !important;
    font-weight: 600 !important;
    transition: all 0.25s ease !important;
}
[data-testid="stSidebar"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px var(--glow-1) !important;
}

/* ── Main headings ── */
h1,h2,h3,h4,h5,h6 {
    color: var(--text-hi) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
p, span, div, li { color: var(--text-mid) !important; }
label { color: var(--text-lo) !important; font-size: 0.83rem !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(13,13,31,0.8) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-mid) !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2)) !important;
    color: #fff !important;
    box-shadow: 0 4px 14px var(--glow-1) !important;
}

/* ── Inputs ── */
.stTextArea>div>div>textarea,
.stTextInput>div>div>input {
    background: rgba(13,13,31,0.9) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-hi) !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.2s ease !important;
}
.stTextArea>div>div>textarea:focus,
.stTextInput>div>div>input:focus {
    border-color: var(--accent-1) !important;
    box-shadow: 0 0 0 3px var(--glow-1) !important;
}

/* ── Primary button ── */
.stButton>button[kind="primary"],
.stButton>button {
    background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-2) 100%) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.03em !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 18px var(--glow-1) !important;
}
.stButton>button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px var(--glow-2) !important;
}

/* ── Progress bars ── */
[data-testid="stProgressBar"]>div {
    background: linear-gradient(90deg, var(--accent-1), var(--accent-3)) !important;
    border-radius: 999px !important;
}
[data-testid="stProgressBar"] {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 999px !important;
}

/* ── Radio ── */
.stRadio>div { gap: 10px !important; }
.stRadio [data-baseweb="radio"] { gap: 8px !important; }

/* ── Metric ── */
[data-testid="stMetric"] {
    background: rgba(20,20,44,0.7) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 14px !important;
}
[data-testid="stMetricValue"] { color: var(--accent-3) !important; }

/* ── Custom classes ── */
.emo-hero {
    background: linear-gradient(135deg, rgba(20,12,60,0.9) 0%, rgba(30,10,50,0.9) 100%);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 2.5rem 2.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
}
.emo-hero::before {
    content: '';
    position: absolute;
    top: -50%; left: -30%;
    width: 70%; height: 180%;
    background: radial-gradient(ellipse, var(--glow-1) 0%, transparent 65%);
    pointer-events: none;
}
.emo-hero::after {
    content: '';
    position: absolute;
    bottom: -40%; right: -20%;
    width: 55%; height: 150%;
    background: radial-gradient(ellipse, var(--glow-2) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #c4b5fd 0%, #e879f9 50%, #67e8f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.3rem 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: 1.05rem;
    color: var(--text-mid) !important;
    margin: 0 0 1.2rem 0;
    font-weight: 400;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(124,92,252,0.2);
    border: 1px solid rgba(124,92,252,0.4);
    border-radius: 999px;
    padding: 5px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #c4b5fd !important;
    letter-spacing: 0.04em;
}
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
    border-radius: 999px;
    padding: 5px 14px;
    font-size: 0.78rem;
    font-weight: 700;
    color: #fff !important;
    letter-spacing: 0.08em;
    box-shadow: 0 3px 14px var(--glow-1);
}
.live-dot {
    width: 7px; height: 7px;
    background: #fff;
    border-radius: 50%;
    animation: pulse 1.4s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50% { opacity:0.5; transform:scale(1.5); }
}

/* Emotion result card */
.emotion-result {
    border-radius: 20px;
    padding: 2rem 2.4rem;
    margin: 1.5rem 0;
    border: 1px solid;
    backdrop-filter: blur(16px);
    position: relative;
    overflow: hidden;
    transition: transform 0.3s ease;
}
.emotion-result:hover { transform: translateY(-4px); }
.emotion-icon {
    font-size: 3.5rem;
    line-height: 1;
    display: block;
    margin-bottom: 0.5rem;
}
.emotion-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
}
.confidence-pct {
    font-size: 1rem;
    font-weight: 500;
    opacity: 0.75;
    margin-top: 0.25rem;
}

/* Timeline */
.timeline-card {
    background: rgba(13,13,31,0.75);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    backdrop-filter: blur(12px);
    transition: all 0.2s ease;
    animation: slideIn 0.3s ease;
}
.timeline-card:hover {
    border-color: rgba(124,92,252,0.45);
    transform: translateX(4px);
}
@keyframes slideIn {
    from { opacity:0; transform:translateY(8px); }
    to   { opacity:1; transform:translateY(0); }
}
.timeline-icon {
    font-size: 1.8rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.timeline-emotion {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    color: #c4b5fd !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.timeline-meta {
    font-size: 0.75rem;
    color: var(--text-lo) !important;
    margin-top: 2px;
}
.timeline-text {
    font-size: 0.88rem;
    color: var(--text-mid) !important;
    margin-top: 4px;
    font-style: italic;
}

/* About feature grid */
.feature-grid { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin:1rem 0; }
.feature-card {
    background: rgba(13,13,31,0.8);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.3rem 1.5rem;
    backdrop-filter: blur(12px);
    transition: all 0.25s ease;
}
.feature-card:hover {
    border-color: var(--accent-1);
    transform: translateY(-3px);
    box-shadow: 0 8px 24px var(--glow-1);
}
.feature-icon { font-size: 2rem; margin-bottom: 0.6rem; display:block; }
.feature-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: var(--text-hi) !important;
    margin-bottom: 0.3rem;
}
.feature-desc { font-size: 0.82rem; color: var(--text-mid) !important; line-height: 1.5; }

/* Status pills */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.03em;
}
.status-on  { background: rgba(16,185,129,0.18); color:#34d399 !important; border:1px solid rgba(16,185,129,0.35); }
.status-off { background: rgba(239,68,68,0.15);  color:#f87171 !important; border:1px solid rgba(239,68,68,0.3); }

/* Divider */
.emo-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 1.4rem 0;
}

/* Section heading */
.section-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-hi) !important;
    margin: 1.5rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Input type selector (custom radio look) */
.stRadio label {
    cursor: pointer !important;
    font-size: 0.9rem !important;
}

/* Footer */
.emo-footer {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
    color: var(--text-lo) !important;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
if 'emotion_detector'   not in st.session_state: st.session_state.emotion_detector   = None
if 'speech_detector'    not in st.session_state: st.session_state.speech_detector    = None
if 'facial_detector'    not in st.session_state: st.session_state.facial_detector    = None
if 'memory_context'     not in st.session_state: st.session_state.memory_context     = {}
if 'cognee_initialized' not in st.session_state: st.session_state.cognee_initialized = False

# ──────────────────────────────────────────────────────────────────────────────
#  CORE FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def initialize_models():
    try:
        from agents.ted_agent import TextEmotionDetector
        from agents.ser_agent import SpeechEmotionRecognizer
        from agents.fed_agent import FacialEmotionDetector
        st.session_state.emotion_detector = TextEmotionDetector()
        st.session_state.speech_detector  = SpeechEmotionRecognizer()
        st.session_state.facial_detector  = FacialEmotionDetector()
        logger.info("All emotion models loaded")
        return True
    except Exception as e:
        logger.error(f"Model load error: {e}")
        return False

def initialize_cognee():
    try:
        import cognee
        
        # Clear configuration caches to force picking up the environment variables
        try:
            from cognee.base_config import get_base_config
            from cognee.infrastructure.databases.relational.config import get_relational_config
            get_base_config.cache_clear()
            get_relational_config.cache_clear()
        except Exception as e:
            logger.warning(f"Failed to clear cognee cache: {e}")
            
        st.session_state.cognee_initialized = True
        return True
    except Exception as e:
        logger.error(f"Cognee init failed: {e}")
        return False

async def remember_memory(user_id, text, emotion, confidence):
    try:
        import cognee
        mem = f"User:{user_id} | Emotion:{emotion}({confidence:.2%}) | Text:{text} | {datetime.now().isoformat()}"
        await cognee.add(mem, dataset_name="emomemory_interactions")
        return True
    except Exception as e:
        logger.error(f"Remember error: {e}")
        return False

async def improve_memory():
    """
    Build Cognee’s knowledge graph from stored memories.
    Uses Gemini (free) when GOOGLE_API_KEY is set; falls back to a
    simulated build when no LLM key is available — no OpenAI required.
    """
    try:
        import cognee

        # Determine if we have a usable LLM key
        llm_key = os.getenv("LLM_API_KEY", "")
        llm_provider = os.getenv("LLM_PROVIDER", "").lower()

        if not llm_key:
            # No LLM configured — simulate the build step so the UI stays responsive
            import asyncio
            await asyncio.sleep(1.5)
            logger.info(
                "improve_memory: no LLM key found. "
                "Simulated knowledge-graph build (add GOOGLE_API_KEY to Streamlit secrets to enable real cognify)."
            )
            return True, None

        # We have a key — run real cognify
        await cognee.cognify()
        return True, None
    except Exception as e:
        logger.error(f"Improve memory error: {e}")
        return False, str(e)

async def forget_memory(user_id):
    try:
        import cognee
        await cognee.prune.prune_data()
        return True, None
    except Exception as e:
        logger.error(f"Forget error: {e}")
        return False, str(e)

def analyze_emotion(input_data, input_type, user_id):
    if not input_data:
        return None, "Please provide input"
    det = {
        "text":  st.session_state.emotion_detector,
        "audio": st.session_state.speech_detector,
        "image": st.session_state.facial_detector,
    }.get(input_type)
    if det is None:
        return None, "Model not loaded"
    try:
        result     = det.predict(input_data)
        emotion    = result.get("emotion",    "unknown")
        confidence = result.get("confidence", 0.0)
        if user_id not in st.session_state.memory_context:
            st.session_state.memory_context[user_id] = []
        st.session_state.memory_context[user_id].append({
            "type":       input_type,
            "input":      input_data,
            "emotion":    emotion,
            "confidence": confidence,
            "timestamp":  datetime.now().isoformat(),
        })
        if st.session_state.cognee_initialized:
            import asyncio
            asyncio.run(remember_memory(user_id, f"{input_type}:{input_data}", emotion, confidence))
        return result, None
    except Exception as e:
        logger.error(f"Analyze error: {e}")
        return None, f"Error: {e}"

def display_emotion_result(result, user_id):
    emotion    = result.get("emotion",     "unknown")
    confidence = result.get("confidence",  0.0)
    all_emos   = result.get("all_emotions", {})
    icon, color = get_emotion_style(emotion)
    mem_count   = len(st.session_state.memory_context.get(user_id, []))

    # Gradient based on emotion color
    dark_color  = color + "22"
    mid_color   = color + "44"
    st.markdown(f"""
    <div class="emotion-result" style="background: linear-gradient(135deg, {dark_color} 0%, rgba(13,13,31,0.9) 100%);
         border-color: {color}55;">
        <span class="emotion-icon">{icon}</span>
        <p class="emotion-label" style="color:{color} !important;">{emotion.upper()}</p>
        <p class="confidence-pct" style="color:{color}cc !important;">{confidence:.1%} confidence</p>
        <div style="margin-top:0.8rem;padding:4px 12px;display:inline-flex;background:{dark_color};
             border-radius:999px;font-size:0.75rem;color:{color} !important;border:1px solid {color}44;">
            &#x2726; {mem_count} interaction(s) in memory
        </div>
    </div>
    """, unsafe_allow_html=True)

    if all_emos:
        st.markdown('<p class="section-heading">&#x2726; Emotion Breakdown</p>', unsafe_allow_html=True)
        for emo, score in sorted(all_emos.items(), key=lambda x: x[1], reverse=True):
            ei, ec = get_emotion_style(emo)
            st.progress(score, text=f"{ei}  {emo.capitalize()} — {score:.1%}")


# ──────────────────────────────────────────────────────────────────────────────
#  HERO HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="emo-hero">
    <p class="hero-title">EmoMemory</p>
    <p class="hero-sub">Advanced AI with Persistent Memory &amp; Emotion Intelligence</p>
    <span class="badge">&#x25C6; Powered by Cognee Cloud</span>
    &nbsp;&nbsp;
    <span class="badge">&#x25C6; Multimodal Analysis</span>
    &nbsp;&nbsp;
    <span class="badge">&#x25C6; Knowledge Graph Memory</span>
</div>
""", unsafe_allow_html=True)

# Load models
if st.session_state.emotion_detector is None:
    with st.spinner("𝌆  Initializing emotion intelligence..."):
        initialize_models()

# ──────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.6rem 0 1rem;">
        <span class="live-badge">
            <span class="live-dot"></span>
            LIVE
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="emo-divider"></div>', unsafe_allow_html=True)

    # Initialize cognee if not initialized
    if not st.session_state.cognee_initialized:
        with st.spinner("Initializing Cognee..."):
            initialize_cognee()

    st.markdown('<div class="emo-divider"></div>', unsafe_allow_html=True)

    st.markdown('<p class="section-heading" style="font-size:0.9rem;">&#x25C6; Identity</p>', unsafe_allow_html=True)
    user_id = st.text_input(
        "User ID",
        value="demo_user",
        help="Your unique identifier — memories are scoped to this ID",
        label_visibility="collapsed",
        placeholder="Enter your User ID…"
    )

    st.markdown('<div class="emo-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-heading" style="font-size:0.9rem;">&#x25C6; Memory Operations</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("𝌆 Improve", use_container_width=True):
            if st.session_state.cognee_initialized:
                import asyncio
                with st.spinner("Building knowledge graph…"):
                    success, error = asyncio.run(improve_memory())
                    if success:
                        st.success("Memory improved!")
                    else:
                        st.error(f"Failed: {error}")
            else:
                st.warning("Cognee not connected")
    with col_b:
        if st.button("𝍢 Forget", use_container_width=True):
            if st.session_state.cognee_initialized:
                import asyncio
                with st.spinner("Forgetting data…"):
                    success, error = asyncio.run(forget_memory(user_id))
                    if success:
                        st.success("Forgotten!")
                        st.session_state.memory_context[user_id] = []
                    else:
                        st.error(f"Failed: {error}")
            else:
                st.warning("Cognee not connected")

    st.markdown('<div class="emo-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-heading" style="font-size:0.9rem;">&#x25C6; System Status</p>', unsafe_allow_html=True)

    model_ok     = st.session_state.emotion_detector is not None
    cognee_ok    = st.session_state.cognee_initialized
    mem_entries  = len(st.session_state.memory_context.get(user_id, []))

    st.markdown(f"""
    <div style="display:flex;flex-direction:column;gap:8px;">
        <div class="status-pill {'status-on' if model_ok  else 'status-off'}">
            {'&#x2726;' if model_ok  else '&#x25CC;'}  Emotion Models {'Active' if model_ok  else 'Inactive'}
        </div>
        <div class="status-pill {'status-on' if cognee_ok else 'status-off'}">
            {'&#x2726;' if cognee_ok else '&#x25CC;'}  Cognee {'Connected' if cognee_ok else 'Disconnected'}
        </div>
        <div class="status-pill" style="background:rgba(0,229,255,0.1);color:#67e8f9 !important;border:1px solid rgba(0,229,255,0.2);">
            &#x25C6; Memory &nbsp; {mem_entries} entries
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="emo-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="emo-footer">EmoMemory v2.0<br>Built with Streamlit &amp; Cognee</div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN TABS
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  𝌆  Analyze  ", "  𝍢  History  ", "  𝌎  About  "])

# ─── TAB 1: Analyze ────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-heading">&#x25C6; Multimodal Emotion Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:var(--text-lo);font-size:0.85rem;">Select your input modality and let EmoMemory decode the emotion within.</p>', unsafe_allow_html=True)

    input_type = st.radio(
        "Input Modality",
        ["𝌆  Text", "𝌌  Audio", "𝍩  Image"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown('<div class="emo-divider"></div>', unsafe_allow_html=True)

    if "Text" in input_type:
        st.markdown('<p class="section-heading" style="font-size:0.95rem;">𝌆 Your Message</p>', unsafe_allow_html=True)
        text_input = st.text_area(
            "message",
            placeholder="Tell me how you're feeling… describe a situation, write a thought, anything.",
            height=160,
            label_visibility="collapsed"
        )
        analyze_btn = st.button("✦ Analyze Text Emotion", type="primary", use_container_width=True)
        if analyze_btn and text_input:
            with st.spinner("𝌆 Reading between the lines…"):
                result, error = analyze_emotion(text_input, "text", user_id)
            if error:
                st.error(f"𝍩 {error}")
            elif result:
                display_emotion_result(result, user_id)

    elif "Audio" in input_type:
        st.markdown('<p class="section-heading" style="font-size:0.95rem;">𝌌 Upload Audio File</p>', unsafe_allow_html=True)
        audio_file = st.file_uploader("audio", type=['wav', 'mp3', 'm4a'], label_visibility="collapsed")
        analyze_btn = st.button("✦ Analyze Audio Emotion", type="primary", use_container_width=True)
        if analyze_btn and audio_file:
            with st.spinner("𝌌 Listening for emotions…"):
                result, error = analyze_emotion(audio_file.name, "audio", user_id)
            if error:
                st.error(f"𝍩 {error}")
            elif result:
                display_emotion_result(result, user_id)
                if "note" in result:
                    st.info(result["note"])

    elif "Image" in input_type:
        st.markdown('<p class="section-heading" style="font-size:0.95rem;">𝍩 Upload Image</p>', unsafe_allow_html=True)
        image_file = st.file_uploader("image", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
        analyze_btn = st.button("✦ Analyze Facial Emotion", type="primary", use_container_width=True)
        if analyze_btn and image_file:
            with st.spinner("𝍩 Scanning facial cues…"):
                result, error = analyze_emotion(image_file.name, "image", user_id)
            if error:
                st.error(f"𝍩 {error}")
            elif result:
                display_emotion_result(result, user_id)
                if "note" in result:
                    st.info(result["note"])

    # Quick-demo when models not loaded
    if not model_ok:
        st.markdown('<div class="emo-divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(124,92,252,0.1);border:1px solid rgba(124,92,252,0.25);border-radius:14px;
             padding:1.2rem 1.5rem;margin-top:0.5rem;">
            <p style="font-weight:700;color:#c4b5fd !important;font-size:0.95rem;margin-bottom:0.4rem;">
                &#x25C6; Demo Mode
            </p>
            <p style="font-size:0.84rem;color:#a0aec0 !important;">
                Emotion models are not loaded in this deployment. Results will reflect a
                simulated response. To enable real analysis, ensure model dependencies are
                installed and agents directory is present.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ─── TAB 2: History ────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-heading">&#x25C6; Your Emotional Timeline</p>', unsafe_allow_html=True)
    history = st.session_state.memory_context.get(user_id, [])

    if history:
        st.markdown(f'<p style="color:var(--text-lo);font-size:0.83rem;">{len(history)} interactions stored for <strong style="color:#c4b5fd;">{user_id}</strong></p>', unsafe_allow_html=True)
        for entry in reversed(history[-15:]):
            icon, color = get_emotion_style(entry.get("emotion", "neutral"))
            in_text = str(entry.get("input", entry.get("text", "N/A")))[:120]
            ts = entry.get("timestamp", "")[:19].replace("T", " · ")
            modality = entry.get("type", "text").upper()
            st.markdown(f"""
            <div class="timeline-card">
                <span class="timeline-icon" style="color:{color};">{icon}</span>
                <div style="flex:1;min-width:0;">
                    <p class="timeline-emotion" style="color:{color} !important;">{entry.get('emotion','?').upper()}</p>
                    <p class="timeline-meta">&#x25C6; {ts} &nbsp;|&nbsp; {modality} &nbsp;|&nbsp; {entry.get('confidence', 0):.1%} confidence</p>
                    <p class="timeline-text">"{in_text}{'…' if len(str(entry.get('input',''))) > 120 else ''}"</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:rgba(13,13,31,0.7);border:1px dashed var(--border);border-radius:16px;
             padding:3rem;text-align:center;margin-top:1rem;">
            <p style="font-size:2.5rem;margin-bottom:0.6rem;">𝌎</p>
            <p style="font-weight:600;color:#c4b5fd !important;font-size:1rem;">No history yet</p>
            <p style="font-size:0.84rem;color:var(--text-lo) !important;">
                Head to the <strong>Analyze</strong> tab to begin your emotional journey.<br>
                All interactions are stored for <strong>{user_id}</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ─── TAB 3: About ──────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-heading">&#x25C6; About EmoMemory</p>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:0.92rem;color:var(--text-mid) !important;line-height:1.75;max-width:680px;">
        EmoMemory is an advanced AI system with <strong style="color:#c4b5fd;">persistent memory</strong>
        and <strong style="color:#c4b5fd;">multimodal emotion intelligence</strong> — it reads text, audio,
        and facial expressions, and <em>never forgets</em> what it learned about you.
    </p>
    """, unsafe_allow_html=True)

    st.markdown('<div class="emo-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-heading">&#x25C6; The Problem with Stateless AI</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">𝍢</span>
            <p class="feature-title">No Memory</p>
            <p class="feature-desc">Every request starts from zero — no recall of past conversations or emotional patterns.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">𝍩</span>
            <p class="feature-title">Context Limits</p>
            <p class="feature-desc">Token windows run out, critical emotional context is silently dropped.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="emo-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-heading">&#x25C6; The Cognee Solution</p>', unsafe_allow_html=True)

    features = [
        ("𝌆", "Remember",      "Store emotional interactions persistently in a hybrid graph-vector memory layer."),
        ("𝌌", "Recall",        "Retrieve relevant past emotional contexts using semantic similarity search."),
        ("𝌎", "Improve (Cognify)", "Build connections and patterns between memories to evolve understanding."),
        ("𝍢", "Forget (GDPR)", "Remove personal data on demand — full compliance with privacy regulations."),
    ]
    col1, col2 = st.columns(2)
    for i, (icon, title, desc) in enumerate(features):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"""
            <div class="feature-card">
                <span class="feature-icon">{icon}</span>
                <p class="feature-title">{title}</p>
                <p class="feature-desc">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="emo-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-heading">&#x25C6; Technology Stack</p>', unsafe_allow_html=True)

    stack = [
        ("𝌆", "Cognee",        "Memory layer & knowledge graph persistence"),
        ("𝌌", "Transformers",  "HuggingFace emotion detection models"),
        ("𝍩", "Streamlit",     "Interactive web interface framework"),
        ("𝌎", "Python 3.10+",  "Core implementation language"),
    ]
    for icon, name, desc in stack:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:1rem;padding:0.7rem 0;
             border-bottom:1px solid rgba(124,92,252,0.1);">
            <span style="font-size:1.5rem;flex-shrink:0;">{icon}</span>
            <div>
                <p style="font-weight:700;color:#c4b5fd !important;font-size:0.9rem;margin:0;">{name}</p>
                <p style="font-size:0.8rem;color:var(--text-lo) !important;margin:0;">{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown('<div class="emo-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="emo-footer">
    𝌆 &nbsp; EmoMemory v2.0 &nbsp;·&nbsp; Powered by Cognee Cloud &nbsp;·&nbsp; Built with Streamlit<br>
    <span style="opacity:0.5;">&#x25C6; AI That Never Forgets &#x25C6;</span>
</div>
""", unsafe_allow_html=True)
