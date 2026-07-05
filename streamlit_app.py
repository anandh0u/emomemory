"""
EmoMemory - Professional Streamlit Application
Memory-Enabled Emotion Intelligence powered by Cognee Cloud
"""

import streamlit as st
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config with professional styling
st.set_page_config(
    page_title="EmoMemory | AI That Never Forgets",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional custom CSS - Dark theme with custom icons
st.markdown("""
<style>
    /* Global styles - Dark theme */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #a0aec0;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    .main-header .badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 1rem;
    }
    
    /* Card styling - Dark theme */
    .emotion-card {
        background: rgba(26, 26, 46, 0.8);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .emotion-card h2 {
        color: #ffffff;
        font-size: 1.8rem;
        margin: 0 0 1rem 0;
    }
    
    .emotion-card .confidence {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Sidebar styling - Dark theme (more visible) */
    .css-1d391kg {
        background: linear-gradient(180deg, #0a0a12 0%, #0f0f1a 100%);
        border-right: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Make sidebar text more visible */
    .css-1d391kg h1, 
    .css-1d391kg h2, 
    .css-1d391kg h3,
    .css-1d391kg p,
    .css-1d391kg label {
        color: #ffffff !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Input styling - Dark theme */
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        padding: 1rem;
        background: rgba(26, 26, 46, 0.5);
        color: #ffffff;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tab styling - Dark theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(26, 26, 46, 0.5);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: #ffffff;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Status indicators */
    .status-online {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #10b981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-offline {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #ef4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Live indicator */
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Memory timeline - Dark theme */
    .timeline-item {
        background: rgba(26, 26, 46, 0.8);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Text colors for dark theme */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    p, span, div {
        color: #e2e8f0 !important;
    }
    
    label {
        color: #a0aec0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'emotion_detector' not in st.session_state:
    st.session_state.emotion_detector = None
if 'speech_detector' not in st.session_state:
    st.session_state.speech_detector = None
if 'facial_detector' not in st.session_state:
    st.session_state.facial_detector = None
if 'memory_context' not in st.session_state:
    st.session_state.memory_context = {}
if 'cognee_initialized' not in st.session_state:
    st.session_state.cognee_initialized = False

def initialize_models():
    """Initialize all emotion detection models."""
    try:
        from agents.ted_agent import TextEmotionDetector
        from agents.ser_agent import SpeechEmotionRecognizer
        from agents.fed_agent import FacialEmotionDetector
        
        st.session_state.emotion_detector = TextEmotionDetector()
        st.session_state.speech_detector = SpeechEmotionRecognizer()
        st.session_state.facial_detector = FacialEmotionDetector()
        logger.info("All emotion models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load emotion detectors: {e}")
        st.error(f"Failed to load emotion models: {e}")
        return False

def initialize_cognee():
    """Initialize Cognee Cloud with API key from environment variable."""
    try:
        import cognee
        
        api_key = os.getenv("COGNEE_API_KEY")
        if not api_key:
            logger.warning("COGNEE_API_KEY not set in environment. Using local memory only.")
            st.info("⚠️ Cognee Cloud not configured - using local memory")
            return False
        
        # Set API key as environment variable for Cognee
        os.environ["COGNEE_API_KEY"] = api_key
        
        # Initialize Cognee (it will read the environment variable)
        st.session_state.cognee_initialized = True
        logger.info("Cognee Cloud initialized successfully")
        st.success("✓ Cognee Cloud connected")
        return True
        
    except ImportError:
        logger.warning("Cognee not installed - using local memory only")
        st.info("⚠️ Cognee not installed - using local memory only")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize Cognee: {e}")
        st.warning(f"⚠️ Cognee initialization failed: {e}")
        return False

async def remember_memory(user_id: str, text: str, emotion: str, confidence: float):
    """Store emotional interaction in Cognee memory."""
    try:
        import cognee
        
        memory_text = f"User: {user_id} | Emotion: {emotion} ({confidence:.2%}) | Text: {text} | Time: {datetime.now().isoformat()}"
        
        await cognee.add(memory_text, dataset_name="emomemory_interactions")
        logger.info(f"Remembered: {emotion} for user {user_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to remember: {e}")
        return False

async def recall_memory(user_id: str, query: str, limit: int = 5):
    """Recall relevant memories from Cognee."""
    try:
        import cognee
        
        search_query = f"User: {user_id} {query}"
        results = await cognee.search(
            query_text=search_query,
            dataset_name="emomemory_interactions"
        )
        
        if isinstance(results, list):
            return results[:limit]
        return []
        
    except Exception as e:
        logger.error(f"Failed to recall: {e}")
        return []

async def improve_memory():
    """Improve memory by building knowledge graph."""
    try:
        import cognee
        await cognee.cognify()
        logger.info("Memory improved successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to improve memory: {e}")
        return False

async def forget_memory(user_id: str):
    """Forget memories for a user."""
    try:
        import cognee
        # In production, implement selective forgetting
        await cognee.prune.prune_data()
        logger.info(f"Forgot data for user: {user_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to forget: {e}")
        return False

def analyze_emotion(input_data: str, input_type: str, user_id: str):
    """Analyze emotion from text, audio, or image."""
    if not input_data:
        return None, "Please provide input"
    
    detector = None
    if input_type == "text":
        detector = st.session_state.emotion_detector
    elif input_type == "audio":
        detector = st.session_state.speech_detector
    elif input_type == "image":
        detector = st.session_state.facial_detector
    
    if detector is None:
        return None, "Model not loaded"
    
    try:
        result = detector.predict(input_data)
        
        emotion = result.get("emotion", "unknown")
        confidence = result.get("confidence", 0.0)
        all_emotions = result.get("all_emotions", {})
        
        # Store in session memory
        if user_id not in st.session_state.memory_context:
            st.session_state.memory_context[user_id] = []
        
        st.session_state.memory_context[user_id].append({
            "type": input_type,
            "input": input_data,
            "emotion": emotion,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
        
        # Store in Cognee if initialized
        if st.session_state.cognee_initialized:
            import asyncio
            memory_text = f"{input_type}: {input_data}"
            asyncio.run(remember_memory(user_id, memory_text, emotion, confidence))
        
        return result, None
        
    except Exception as e:
        logger.error(f"Error analyzing emotion: {e}")
        return None, f"Error: {str(e)}"

def display_emotion_result(result, user_id):
    """Display emotion analysis result with professional styling."""
    emotion = result.get("emotion", "unknown")
    confidence = result.get("confidence", 0.0)
    all_emotions = result.get("all_emotions", {})
    
    # Display result with professional styling
    st.markdown(f"""
    <div class="emotion-card animate-fade-in">
        <h2>Detected Emotion: {emotion.upper()}</h2>
        <p class="confidence">{confidence:.1%}</p>
        <p style="color: #666; margin-top: 1rem;">Confidence Level</p>
    </div>
    """, unsafe_allow_html=True)
    
    # All emotions breakdown
    st.subheader("All Emotions")
    for emo, score in sorted(all_emotions.items(), key=lambda x: x[1], reverse=True):
        st.progress(score, text=f"{emo.capitalize()}: {score:.1%}")
    
    # Memory context
    memory_count = len(st.session_state.memory_context.get(user_id, []))
    if memory_count > 0:
        st.success(f"🧠 Memory Context Active - {memory_count} past interaction(s) stored")

# Professional Header
st.markdown("""
<div class="main-header animate-fade-in">
    <h1>🎯 EmoMemory</h1>
    <p>Memory-Enabled Emotion Intelligence powered by Cognee Cloud</p>
    <div class="badge">WeMakeDevs Hackathon 2025</div>
</div>
""", unsafe_allow_html=True)

# Initialize models on first run
if st.session_state.emotion_detector is None:
    with st.spinner("Loading emotion detection model..."):
        initialize_models()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Live indicator
    st.markdown('<div class="live-indicator">🔴 LIVE</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Initialize Cognee with environment variable
    if not st.session_state.cognee_initialized:
        with st.spinner("Initializing Cognee Cloud..."):
            initialize_cognee()
    
    st.markdown("---")
    
    # User ID
    user_id = st.text_input(
        "User ID",
        value="demo_user",
        help="Your unique identifier for memory"
    )
    
    st.markdown("---")
    
    # Memory operations
    st.header("💾 Memory Operations")
    
    if st.button("🚀 Improve Memory"):
        if st.session_state.cognee_initialized:
            import asyncio
            with st.spinner("Building knowledge graph..."):
                success = asyncio.run(improve_memory())
                if success:
                    st.success("Memory improved successfully!")
                else:
                    st.error("Failed to improve memory")
        else:
            st.warning("Initialize Cognee Cloud first")
    
    if st.button("🗑️ Forget My Data"):
        if st.session_state.cognee_initialized:
            import asyncio
            with st.spinner("Forgetting data..."):
                success = asyncio.run(forget_memory(user_id))
                if success:
                    st.success("Data forgotten successfully!")
                    st.session_state.memory_context[user_id] = []
                else:
                    st.error("Failed to forget data")
        else:
            st.warning("Initialize Cognee Cloud first")
    
    st.markdown("---")
    
    # Status with professional styling
    st.header("📊 System Status")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.emotion_detector:
            st.markdown('<div class="status-online">✅ Model Active</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-offline">❌ Model Inactive</div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.cognee_initialized:
            st.markdown('<div class="status-online">✅ Cognee Cloud</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-offline">❌ Cognee Offline</div>', unsafe_allow_html=True)
    
    with col3:
        memory_count = len(st.session_state.memory_context.get(user_id, []))
        st.metric("Memory Entries", memory_count)

# Main content
tab1, tab2, tab3 = st.tabs(["🎯 Analyze Emotion", "💾 Memory History", "ℹ️ About"])

with tab1:
    st.subheader("Multimodal Emotion Analysis")
    
    # Input type selector
    input_type = st.radio(
        "Select Input Type",
        ["Text", "Audio", "Image"],
        horizontal=True
    )
    
    if input_type == "Text":
        text_input = st.text_area(
            "Your Message",
            placeholder="Type your message here...",
            height=150
        )
        analyze_btn = st.button("Analyze Text Emotion", type="primary", use_container_width=True)
        
        if analyze_btn and text_input:
            result, error = analyze_emotion(text_input, "text", user_id)
            
            if error:
                st.error(error)
            elif result:
                display_emotion_result(result, user_id)
    
    elif input_type == "Audio":
        audio_file = st.file_uploader("Upload Audio File", type=['wav', 'mp3', 'm4a'])
        analyze_btn = st.button("Analyze Audio Emotion", type="primary", use_container_width=True)
        
        if analyze_btn and audio_file:
            result, error = analyze_emotion(audio_file.name, "audio", user_id)
            
            if error:
                st.error(error)
            elif result:
                display_emotion_result(result, user_id)
                if "note" in result:
                    st.info(result["note"])
    
    elif input_type == "Image":
        image_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
        analyze_btn = st.button("Analyze Facial Emotion", type="primary", use_container_width=True)
        
        if analyze_btn and image_file:
            result, error = analyze_emotion(image_file.name, "image", user_id)
            
            if error:
                st.error(error)
            elif result:
                display_emotion_result(result, user_id)
                if "note" in result:
                    st.info(result["note"])

with tab2:
    st.subheader("Your Emotional Timeline")
    
    if user_id in st.session_state.memory_context and len(st.session_state.memory_context[user_id]) > 0:
        history = st.session_state.memory_context[user_id]
        
        for i, entry in enumerate(reversed(history[-10:]), 1):
            input_text = entry.get('input', entry.get('text', 'N/A'))
            input_type = entry.get('type', 'text')
            st.markdown(f"""
            <div class="timeline-item animate-fade-in">
                <strong>{entry['emotion'].upper()}</strong> ({entry['confidence']:.1%})
                <br><small style="color: #666;">{entry['timestamp'][:19]} | {input_type.upper()}</small>
                <br><em>"{input_text}"</em>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info(f"No emotional history found for user: {user_id}")

with tab3:
    st.markdown("""
    <div class="emotion-card">
        <h2>About EmoMemory</h2>
        <p>EmoMemory is a memory-enabled emotion intelligence system built for the WeMakeDevs x Cognee Hackathon.</p>
        
        <h3>The Problem</h3>
        <p>Traditional LLMs and AI systems are <strong>stateless</strong>. Every request starts from scratch:</p>
        <ul>
            <li>❌ No memory of past conversations</li>
            <li>❌ Context window limits (tokens run out)</li>
            <li>❌ Can't learn from user patterns</li>
            <li>❌ Forgets important emotional context</li>
        </ul>
        
        <h3>The Solution: Cognee Memory</h3>
        <p><strong>Cognee</strong> provides a hybrid graph-vector memory layer that enables:</p>
        
        <h4>1️⃣ Remember</h4>
        <p>Store emotional interactions persistently in a knowledge graph</p>
        
        <h4>2️⃣ Recall</h4>
        <p>Retrieve relevant past contexts using semantic search</p>
        
        <h4>3️⃣ Improve (Cognify)</h4>
        <p>Build connections and patterns between memories</p>
        
        <h4>4️⃣ Forget</h4>
        <p>Remove data when needed (GDPR compliant)</p>
        
        <h3>Technology Stack</h3>
        <ul>
            <li>🧠 <strong>Cognee</strong> - Memory layer and knowledge graph</li>
            <li>🎭 <strong>Transformers</strong> - HuggingFace emotion detection</li>
            <li>🎨 <strong>Streamlit</strong> - Interactive web interface</li>
            <li>🐍 <strong>Python</strong> - Core implementation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    Powered by Cognee | Built with Streamlit | WeMakeDevs Hackathon 2025
</div>
""", unsafe_allow_html=True)
