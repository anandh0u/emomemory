# EmoMemory: AI That Never Forgets

> **Memory-Enabled Emotion Intelligence powered by Cognee Cloud**
> 
> Built for the WeMakeDevs x Cognee Hackathon 2025

[![Cognee](https://img.shields.io/badge/Powered%20by-Cognee-blue)](https://github.com/topoteretes/cognee)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 The Problem

Traditional AI systems suffer from **amnesia**:

```
❌ Every request is stateless
❌ No memory of past conversations  
❌ Context window limits (tokens run out)
❌ Can't learn from user patterns
❌ Forgets important emotional context
```

**It's like having a therapist who forgets everything you said in the last session!**

## 💡 The Solution

**EmoMemory** uses [Cognee](https://github.com/topoteretes/cognee) to give emotion AI a **permanent, hybrid graph-vector memory layer**:

```
✅ REMEMBER - Store emotional interactions persistently
✅ RECALL   - Retrieve relevant past contexts using semantic search
✅ IMPROVE  - Build knowledge graph connections
✅ FORGET   - Remove data when needed (GDPR compliant)
```

---

## 🌟 Key Features

### 1️⃣ Stateful Emotion Detection
Unlike traditional emotion AI that treats each input independently, EmoMemory maintains context across conversations.

### 2️⃣ Multimodal Emotion Intelligence
- 📝 **Text Emotion Detection** - Analyze text messages
- 🎤 **Speech Emotion Recognition** - Detect emotion in voice
- 👤 **Facial Emotion Detection** - Analyze facial expressions

### 3️⃣ Premium Streamlit Interface
Beautiful dark glassmorphism UI with:
- Real-time emotion analysis
- Emotional history timeline
- Memory management operations
- Custom emoji pack for emotions

### 4️⃣ Complete Memory Lifecycle
Demonstrates all four Cognee memory operations:

| Operation | Purpose |
|-----------|---------|
| **Remember** | Store interactions automatically |
| **Recall** | Retrieve relevant past contexts |
| **Improve** | Build knowledge graph connections |
| **Forget** | Remove data (GDPR compliant) |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) Google API key for enhanced features

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/anandh0u/emomemory.git
cd emomemory
```

2. **Create virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the app**
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📖 Usage

### Web Interface

The Streamlit app includes three main tabs:

1. **Analyze Tab** - Perform emotion analysis on text, audio, or images
2. **History Tab** - View your emotional timeline and past interactions
3. **About Tab** - Learn about the project and technology

### Memory Operations

In the sidebar, you can:
- **Improve Memory** - Build knowledge graph from stored memories
- **Forget My Data** - Remove all your data (GDPR compliant)
- **View System Status** - Check model and Cognee connection status

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        User Input                            │
│           (Text, Audio, Image)                               │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Emotion Detection Models (emo1)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Text (TED)  │  │ Speech (SER) │  │ Facial (FED) │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Cognee Memory Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Vector DB  │  │ Knowledge    │  │   Graph DB   │      │
│  │  (Semantic)  │  │   Graph      │  │ (Relations)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

1. **streamlit_app.py** - Main Streamlit application
2. **agents/ted_agent.py** - Text emotion detection model
3. **agents/ser_agent.py** - Speech emotion recognition model
4. **agents/fed_agent.py** - Facial emotion detection model
5. **Cognee Integration** - Memory layer for persistent storage

---

## 🎓 How It Demonstrates Cognee

This project showcases **all four memory lifecycle operations**:

### ✅ 1. Remember
Every emotional interaction is automatically stored in Cognee's hybrid graph-vector memory.

### ✅ 2. Recall
Relevant past contexts are retrieved using semantic search when analyzing new inputs.

### ✅ 3. Improve
Knowledge graph connections are built to enhance memory relationships.

### ✅ 4. Forget
User data can be completely removed on demand (GDPR compliant).

---

## 🎬 Demo Scenarios

### Scenario 1: Emotional Journey Tracking

```
Session 1:
User: "I just got a new job!"
→ Emotion: Happy (95% confidence)
→ Memory: Stored

Session 2 (next day):
User: "I'm nervous about my first day..."
→ Emotion: Anxious (87% confidence)
→ Context: Previous happiness about new job retrieved

Session 3 (week later):
User: "Things are going great!"
→ Emotion: Happy (92% confidence)
→ Pattern: Job journey tracked across time
```

---

## 🎯 Use Cases

### 1. Mental Health & Therapy
- Track emotional patterns over time
- Identify triggers and progress
- Maintain long-term therapeutic relationships

### 2. Customer Support
- Remember past issues and frustrations
- Provide context-aware responses
- Build better customer relationships

### 3. Personal Journaling
- Track daily emotional states
- Identify patterns and trends
- Maintain persistent emotional history

---

## 🔧 Configuration

### Environment Variables

For Streamlit Cloud deployment, add these to your `.streamlit/secrets.toml`:

```toml
# Optional: Google API Key for enhanced features
GOOGLE_API_KEY = "your_google_api_key_here"
```

### Local Development

Create a `.env` file:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

---

## 📈 Performance

### Model Performance
- **Text Emotion**: ~100-300ms per analysis
- **Speech Emotion**: ~500-1000ms per analysis
- **Facial Emotion**: ~300-600ms per analysis

### Memory Operations
- **Remember**: ~50-200ms to store
- **Recall**: ~100-500ms for semantic search
- **Improve**: ~1-5s for knowledge graph building

---

## 🛣️ Roadmap

### Phase 1: Core Features (Current)
- ✅ Cognee memory integration
- ✅ Four lifecycle operations
- ✅ Multimodal emotion detection
- ✅ Premium Streamlit UI

### Phase 2: Enhanced Intelligence
- 🔄 Automatic pattern detection
- 🔄 Predictive emotional modeling
- 🔄 Multi-user relationship graphs

### Phase 3: Production Ready
- 🔄 Authentication & security
- 🔄 API documentation
- 🔄 Monitoring & analytics

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- **[Cognee](https://github.com/topoteretes/cognee)** - For the incredible memory layer
- **WeMakeDevs** - For hosting this hackathon
- **HuggingFace** - For the emotion detection models
- **Streamlit** - For the amazing web framework

---

## 📞 Contact

- **Project**: EmoMemory
- **Built for**: WeMakeDevs x Cognee Hackathon 2025
- **Repository**: [github.com/anandh0u/emomemory](https://github.com/anandh0u/emomemory)
- **Live Demo**: [emomemory.streamlit.app](https://emomemory.streamlit.app)

---

## 📚 Additional Resources

- [Cognee Documentation](https://docs.cognee.ai)
- [Cognee GitHub](https://github.com/topoteretes/cognee)
- [Hackathon Details](https://hackathon.cognee.ai)
- [WeMakeDevs Community](https://wemakedevs.org)

---

<div align="center">

**Built with ❤️ using Cognee & Streamlit**

*"Making AI that never forgets"*

</div>
