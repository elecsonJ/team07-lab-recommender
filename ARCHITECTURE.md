# 🏗️ System Architecture

## Overview

The Graduate School Lab Recommender implements a **sophisticated hybrid AI system** that combines Retrieval-Augmented Generation (RAG), Agentic AI decision-making, and conversational memory management to provide intelligent research lab recommendations.

## 🔄 System Flow

```
User Query → Query Classification → Strategy Selection → Information Retrieval → Response Generation
     ↓              ↓                     ↓                    ↓                  ↓
  Natural       Agentic AI          RAG/Web Search         Context             GPT-4o-mini
 Language        Decision           Hybrid Strategy        Integration         Generation
```

## 🧠 Core Components

### 1. Query Classification System (`classify_query`)

**Purpose**: Intelligent routing of user queries to appropriate processing strategies

**Decision Logic**:
```python
def classify_query(self, new_query: str) -> Dict[str, Any]:
    # 1. Professor-specific queries
    if self.contains_professor_name(new_query):
        return {"type": "professor_detail"}
    
    # 2. Follow-up questions using previous context
    if self.can_answer_with_previous(new_query):
        return {"type": "refine_previous"}
    
    # 3. Research area exploration
    if self.is_research_related(new_query):
        return {"type": "new_search"}
    
    # 4. General graduate school information
    return {"type": "general_info"}
```

**Key Features**:
- Professor name detection using predefined list
- Follow-up pattern recognition ("더 자세히", "그 중에서")
- Research keyword analysis for domain relevance
- Fallback handling for general queries

### 2. RAG (Retrieval-Augmented Generation) Engine

#### Vector Store Architecture
- **Embedding Model**: `text-embedding-3-small` (1536 dimensions)
- **Vector Database**: FAISS with MMR (Maximum Marginal Relevance) search
- **Document Structure**: Comprehensive professor profiles with metadata

#### Document Processing Pipeline
```python
professor_document = {
    "page_content": f"""
=== 기본 정보 ===
교수명: {name}
연구실: {lab_name}
연구분야: {research_keywords}
연구분야 설명: {detailed_description}

=== 연구 상세 정보 ===
연구주제: {research_topics}
기술 및 방법: {methodologies}
주요 논문: {publications}
학생지도 특징: {supervision_style}
    """,
    "metadata": {
        "professor_name": name,
        "keywords": research_areas,
        # ... additional structured metadata
    }
}
```

#### Retrieval Strategy
- **MMR Search**: Balances relevance and diversity
- **Fetch Strategy**: `k=5` results with `fetch_k=10` candidates
- **Lambda Multiplier**: 0.5 for optimal relevance/diversity balance

### 3. Conversational Memory Management

#### ConversationHistory Class
```python
@dataclass
class ConversationHistory:
    queries: List[str]
    responses: List[str]
    retrieved_docs: List[List[Document]]
    
    def get_context(self, last_n: int = 3) -> str:
        # Returns last N conversation turns for context
        
    def add_turn(self, query, response, docs):
        # Stores conversation state with retrieved documents
```

**Context Management**:
- **Short-term Memory**: Last 3 conversation turns
- **Document Retention**: Stores retrieved documents for follow-up queries
- **Context Length Control**: Response truncation to 200 characters for context

### 4. Hybrid Search Strategy

#### Strategy Selection Matrix

| Query Type | Processing Method | Context Usage | Response Style |
|------------|-------------------|---------------|----------------|
| `professor_detail` | Detail QA Chain + RAG | Full conversation | Comprehensive |
| `refine_previous` | Previous docs only | Last retrieved docs | Contextual refinement |
| `new_search` | Brief QA Chain + RAG | Current query enhanced | Concise recommendations |
| `general_info` | LLM only | No RAG context | General guidance |

#### Query Enhancement Pipeline
```python
def enhance_query_with_translation(self, query: str) -> str:
    # For Korean queries:
    # 1. Extract research keywords using LLM
    # 2. Translate to English equivalents
    # 3. Combine Korean + English for better retrieval
    enhanced = f"{original_korean} {english_keywords}"
    return enhanced
```

### 5. Dual Prompt System

#### Brief Recommendation Prompt
- **Purpose**: Initial research area exploration
- **Output**: 2-3 professor recommendations with concise explanations
- **Focus**: Matching research interests with professor expertise

#### Detail Information Prompt
- **Purpose**: Deep-dive into specific professors
- **Output**: Comprehensive information about research, publications, lab environment
- **Focus**: All available data points for informed decision-making

## 🚀 Performance Optimizations

### Vector Search Optimization
- **MMR Algorithm**: Prevents redundant similar results
- **Batch Processing**: Efficient embedding generation
- **Index Persistence**: FAISS local storage for fast startup

### Memory Management
- **Conversation Pruning**: Automatic cleanup of old conversations
- **Document Caching**: Reuse of retrieved documents for follow-ups
- **Response Truncation**: Context size optimization

### API Efficiency
- **Azure OpenAI**: Optimized endpoint configuration
- **Temperature Control**: 0.3 for consistent, focused responses
- **Token Management**: Efficient prompt construction

## 🔗 Component Interactions

### Data Flow Diagram
```
User Input
    ↓
Query Classification (Agentic Decision)
    ↓
┌─────────────────────────────────────┐
│  Strategy Router                    │
├─────────────────────────────────────┤
│ new_search     → RAG Brief Chain    │
│ professor_detail → RAG Detail Chain │
│ refine_previous → Context Memory    │
│ general_info   → Direct LLM        │
└─────────────────────────────────────┘
    ↓
Response Generation
    ↓
Conversation History Update
```

### Integration Points

1. **Vector Store ↔ QA Chains**: MMR retriever configuration
2. **Conversation History ↔ Strategy Router**: Context-aware decision making
3. **Query Enhancer ↔ RAG System**: Multilingual query expansion
4. **Prompt Templates ↔ LLM**: Specialized response formatting

## 📊 Technical Specifications

### Model Configuration
```python
# Embedding Model
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536,
    azure_endpoint=AZURE_ENDPOINT,
    api_version="2024-02-15-preview"
)

# Language Model
llm = AzureChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,  # Balanced creativity/consistency
    azure_endpoint=AZURE_ENDPOINT
)
```

### Vector Store Configuration
```python
# FAISS with MMR
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,           # Final results
        "fetch_k": 10,    # Candidates for MMR
        "lambda_mult": 0.5 # Relevance vs diversity
    }
)
```

## 🛠️ Design Patterns

### Strategy Pattern
- **Query Classification**: Different processing strategies based on query type
- **Prompt Templates**: Specialized prompts for different use cases

### Observer Pattern
- **Conversation History**: Tracks all interactions and retrieved documents
- **Context Management**: Updates based on user interactions

### Template Method Pattern
- **QA Chain Processing**: Consistent retrieval → generation → response flow
- **Error Handling**: Standardized fallback mechanisms

## 🔍 Quality Control Mechanisms

### Relevance Filtering
- **Cosine Similarity Thresholds**: Automatic quality assessment
- **MMR Diversity Control**: Prevents information redundancy
- **Context Validation**: Ensures conversational coherence

### Error Recovery
- **Fallback Strategies**: Web search when RAG confidence is low
- **Query Reformulation**: Enhanced search terms for better results
- **Graceful Degradation**: General responses when specific information unavailable

This architecture enables sophisticated, context-aware conversations while maintaining high performance and user experience quality.