# 🎓 Graduate School Lab Recommender AI

An intelligent graduate school laboratory recommendation system powered by **RAG (Retrieval-Augmented Generation)** technology and **Agentic AI** for adaptive query processing.

## 🏗️ System Architecture

This system implements a sophisticated **hybrid search strategy** that combines multiple AI technologies:

```
User Query → Query Classification → RAG Search → Quality Assessment → Web Search (Fallback) → Response Generation
```

### Core Components

- **RAG Engine**: Vector-based similarity search using professor research profiles
- **Agentic AI**: Adaptive strategy selection based on query type and result quality
- **Conversational Memory**: Context-aware dialogue management for follow-up questions
- **Hybrid Search**: RAG-first approach with intelligent web search fallback

## 🔍 Technical Implementation

### RAG System
- **Vector Embeddings**: `text-embedding-3-small` (1536 dimensions)
- **Similarity Search**: Cosine similarity matching against professor profiles
- **Vector Database**: Chroma/FAISS for efficient retrieval
- **Threshold-based Quality Control**: Automatic fallback when similarity scores are low

### Agentic AI Strategy
The system intelligently selects search strategies based on:
- Query type classification (research area vs. general questions)
- RAG result confidence scores
- Conversational context and user intent

### Conversational AI
- **Context Management**: Session-based conversation history
- **Follow-up Processing**: Natural dialogue flow with memory retention
- **Response Generation**: GPT-4o-mini with context-aware prompting

## 📊 Dataset & Performance

- **Professor Profiles**: 31 SNU Medical School faculty members
- **Research Papers**: 96 publications indexed
- **Data Completeness**: 95.8%
- **Embedding Dimensions**: 1536 (OpenAI text-embedding-3-small)
- **Response Time**: < 3 seconds average

## 🚀 Key Features

- **🧠 Intelligent Query Processing**: Automatic classification and routing of user queries
- **🔍 Hybrid Search Strategy**: RAG + Web search for comprehensive coverage
- **💬 Natural Conversation**: Context-aware dialogue with memory
- **⚡ Fast Response**: Optimized vector search with quality thresholds
- **🎯 Personalized Recommendations**: Matching based on research interests and preferences

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Backend Framework** | LangChain |
| **LLM** | GPT-4o-mini (Azure OpenAI) |
| **Embeddings** | text-embedding-3-small |
| **Vector Database** | Chroma |
| **Web Search** | Tavily API |
| **Data Processing** | Python, JSON |

## 📁 Project Structure

```
team07-lab-recommender/
├── streamlit_app.py              # Main web application
├── rag_lab_recommender.py        # Core RAG engine with conversation management
├── generate_embeddings.py        # Vector embedding generation
├── professors_final_complete.json # Professor dataset (31 profiles)
├── requirements.txt              # Python dependencies
├── deployment_guide.md           # Deployment instructions
└── .streamlit/                   # Streamlit configuration
```

### Key Files

- **`streamlit_app.py`**: Web interface with session management and chat UI
- **`rag_lab_recommender.py`**: Core RAG implementation with `ConversationHistory` class
- **`generate_embeddings.py`**: Embedding generation and vector database setup
- **`professors_final_complete.json`**: Curated dataset of professor profiles and research areas

## 🏃‍♂️ Quick Start

### Prerequisites

- Python 3.8+
- Azure OpenAI API access
- Tavily API key (for web search)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/elecsonJ/team07-lab-recommender.git
cd team07-lab-recommender

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your API keys:
# AZURE_OPENAI_ENDPOINT=your_endpoint
# OPENAI_API_KEY=your_api_key
# OPENAI_API_VERSION=your_version
# TAVILY_API_KEY=your_tavily_key

# 5. Generate embeddings (first time only)
python generate_embeddings.py

# 6. Run application
streamlit run streamlit_app.py
```

## 🧪 Usage Examples

### Basic Research Area Query
```
User: "I'm interested in AI and machine learning research"
→ RAG Search → Professor matching → Detailed recommendations
```

### Follow-up Questions
```
User: "Tell me more about Professor Kim's recent work"
→ Conversational context → Targeted information retrieval
```

### General Information Query
```
User: "What's the application process for grad school?"
→ Query classification → Web search fallback → Comprehensive answer
```

## 🎯 Learning Outcomes

This project demonstrates:

- **RAG Implementation**: End-to-end retrieval-augmented generation system
- **Agentic AI Design**: Adaptive strategy selection based on context
- **Vector Database Management**: Efficient similarity search and retrieval
- **Conversational AI**: Context-aware dialogue systems
- **Hybrid Search Architecture**: Combining multiple information sources
- **Production Deployment**: Web application with session management

## 🔧 Development Journey

Built during a hackathon as an exploration of:
- Modern RAG architectures and implementation patterns
- Agentic AI systems with decision-making capabilities
- Integration of multiple AI services (OpenAI, Tavily)
- Real-world data processing and embedding generation
- User experience design for AI-powered applications

## 📝 Technical Notes

- **Quality Control**: Implemented cosine similarity thresholds to ensure relevant results
- **Fallback Strategy**: Web search integration when RAG confidence is low
- **Context Management**: Session-based conversation memory for natural dialogue
- **Data Pipeline**: Structured professor profile processing and embedding generation
- **API Integration**: Azure OpenAI and Tavily API with proper error handling

## 🤝 Contributing

This repository serves as a portfolio demonstration of RAG and Agentic AI implementation. The codebase showcases production-ready patterns for:
- Vector database integration
- Conversational AI systems
- Hybrid search architectures
- Modern web application deployment

---

**Team 07** - Graduate School Lab Recommender System Development Team