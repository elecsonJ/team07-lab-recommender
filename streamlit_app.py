import streamlit as st
import os
import sys
from typing import Dict, List, Any
import time

# 현재 디렉토리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_lab_recommender import LabRecommenderRAG, ConversationHistory

# Streamlit 페이지 설정
st.set_page_config(
    page_title="대학원 연구실 추천 AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #e3f2fd;
        flex-direction: row-reverse;
    }
    .chat-message.assistant {
        background-color: #f5f5f5;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin: 0 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
    }
    .chat-message.user .avatar {
        background-color: #2196f3;
        color: white;
    }
    .chat-message.assistant .avatar {
        background-color: #4caf50;
        color: white;
    }
    .chat-message .content {
        flex: 1;
        padding: 0 0.5rem;
    }
    .classification-info {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 4px solid #ff9800;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 0.3rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .sidebar-info {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitRAGApp:
    def __init__(self):
        self.data_path = "professors_final_complete.json"
        self.rag_system = None
        self.init_rag_system()
    
    def init_rag_system(self):
        """RAG 시스템 초기화"""
        if 'rag_system' not in st.session_state:
            with st.spinner('🔄 RAG 시스템을 초기화하고 있습니다...'):
                try:
                    rag_system = LabRecommenderRAG(self.data_path)
                    
                    # 벡터 저장소 로드
                    if not rag_system.load_vector_store():
                        st.warning("⚠️ 기존 벡터 저장소를 찾을 수 없습니다. 새로 생성합니다...")
                        rag_system.create_vector_store()
                    
                    # QA 체인 설정
                    rag_system.setup_qa_chains(k=5)
                    
                    st.session_state.rag_system = rag_system
                    st.success("✅ RAG 시스템이 성공적으로 초기화되었습니다!")
                    
                except Exception as e:
                    st.error(f"❌ RAG 시스템 초기화 실패: {str(e)}")
                    st.stop()
        
        self.rag_system = st.session_state.rag_system
    
    def init_session_state(self):
        """세션 상태 초기화"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'conversation_count' not in st.session_state:
            st.session_state.conversation_count = 0
    
    def render_chat_message(self, role: str, content: str, classification_info: Dict = None):
        """채팅 메시지 렌더링"""
        avatar = "👤" if role == "user" else "🤖"
        css_class = "user" if role == "user" else "assistant"
        
        st.markdown(f"""
        <div class="chat-message {css_class}">
            <div class="avatar">{avatar}</div>
            <div class="content">
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 분류 정보 표시 (AI 응답에만)
        if role == "assistant" and classification_info:
            st.markdown(f"""
            <div class="classification-info">
                <strong>🤖 질문 분류:</strong> {classification_info.get('type', 'unknown')}<br>
                <strong>📝 이유:</strong> {classification_info.get('reason', '알 수 없음')}
            </div>
            """, unsafe_allow_html=True)
    
    def process_user_input(self, user_input: str):
        """사용자 입력 처리"""
        if not user_input.strip():
            return
        
        # 사용자 메시지 저장
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input,
            "timestamp": time.time()
        })
        
        # RAG 시스템으로 처리
        with st.spinner('🔍 답변을 생성하고 있습니다...'):
            try:
                # 분류 정보 가져오기
                classification = self.rag_system.classify_query(user_input)
                
                # 응답 생성
                response = self.rag_system.process_query(user_input)
                
                # 응답 저장
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "classification": classification,
                    "timestamp": time.time()
                })
                
                st.session_state.conversation_count += 1
                
            except Exception as e:
                st.error(f"❌ 오류가 발생했습니다: {str(e)}")
    
    def render_sidebar(self):
        """사이드바 렌더링"""
        with st.sidebar:
            st.markdown("### 🎓 대학원 연구실 추천 AI")
            
            # 대화 통계
            st.markdown(f"""
            <div class="sidebar-info">
                <strong>📊 대화 통계</strong><br>
                • 총 대화 수: {st.session_state.conversation_count}<br>
                • 현재 메시지: {len(st.session_state.messages)}개
            </div>
            """, unsafe_allow_html=True)
            
            # 대화 초기화 버튼
            if st.button("🔄 대화 초기화", use_container_width=True):
                self.rag_system.conversation_history.clear()
                st.session_state.messages = []
                st.session_state.conversation_count = 0
                st.success("대화가 초기화되었습니다!")
                st.rerun()
            
            
            # 사용법 안내
            st.markdown("""
            ### 📖 사용법
            
            **질문 유형:**
            - 🔍 **연구실 추천**: "AI 연구하고 싶어"
            - 🔄 **추가 질문**: "그 중에서 의료 AI는?"
            - 💬 **일반 질문**: "입학 절차는?"
            
            **명령어:**
            - `clear` 또는 `reset`: 대화 초기화
            - `quit` 또는 `exit`: 종료
            """)
            
            # 데이터 정보
            st.markdown("""
            ### 📚 데이터 정보
            - **교수 수**: 31명
            - **논문 수**: 96편
            - **완성도**: 95.8%
            - **마지막 업데이트**: 2025-07-23
            """)
    
    def run(self):
        """메인 앱 실행"""
        # 세션 상태 초기화
        self.init_session_state()
        
        # 사이드바 렌더링
        self.render_sidebar()
        
        # 메인 헤더
        st.markdown('<h1 class="main-header">🎓 대학원 연구실 추천 AI</h1>', unsafe_allow_html=True)
        
        # 안내 메시지 (첫 방문시)
        if len(st.session_state.messages) == 0:
            st.info("""
            👋 **안녕하세요! 대학원 연구실 추천 AI입니다.**
            
            관심있는 연구 분야나 주제를 자유롭게 입력해주세요. 
            AI가 질문을 분석하여 가장 적합한 연구실을 추천해드립니다.
            
            **예시 질문:**
            - "인공지능과 머신러닝 연구하고 싶어"
            - "세포 기작 연구에 관심있어"
            - "대학원 입학 절차가 궁금해"
            """)
        
        # 채팅 메시지들 표시
        for message in st.session_state.messages:
            self.render_chat_message(
                message["role"], 
                message["content"],
                message.get("classification")
            )
        
        # 사용자 입력 받기
        user_input = st.chat_input("메시지를 입력하세요...")
        
        if user_input:
            # 특수 명령어 처리
            if user_input.lower() in ['clear', 'reset', '초기화', '새로시작']:
                self.rag_system.conversation_history.clear()
                st.session_state.messages = []
                st.session_state.conversation_count = 0
                st.success("🔄 대화가 초기화되었습니다!")
                st.rerun()
            elif user_input.lower() in ['quit', 'exit', '종료', '끝']:
                st.success("👋 이용해 주셔서 감사합니다!")
                st.stop()
            else:
                # 일반 사용자 입력 처리
                self.process_user_input(user_input)
                st.rerun()

# 앱 실행
if __name__ == "__main__":
    app = StreamlitRAGApp()
    app.run()