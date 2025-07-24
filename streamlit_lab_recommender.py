"""
서울대 의대 연구실 추천 웹앱 (Streamlit)
벡터 임베딩 + GPT-4o-mini 기반 추천 시스템
"""

import streamlit as st
import json
import numpy as np
import os
import pickle
from typing import List, Dict, Any, Tuple
from openai import AzureOpenAI
import time

# 페이지 설정
st.set_page_config(
    page_title="🔬 서울대 의대 연구실 추천",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class LabRecommendationSystem:
    def __init__(self):
        self.professors_data = []
        self.professor_embeddings = []
        self.client = None
        self.embedding_model = "text-embedding-3-small"
        
    @st.cache_data
    def load_professor_data(_self):
        """교수진 데이터 로드 (캐시됨)"""
        try:
            with open('professors_final_complete.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            _self.professors_data = data["교수진"]
            return True, f"{len(_self.professors_data)}명 교수 데이터 로드 완료"
        except FileNotFoundError:
            return False, "professors_final_complete.json 파일을 찾을 수 없습니다."
        except Exception as e:
            return False, f"데이터 로드 실패: {str(e)}"
    
    @st.cache_data
    def load_embeddings(_self):
        """저장된 임베딩 벡터 로드 (캐시됨)"""
        try:
            with open('professor_embeddings.pkl', 'rb') as f:
                embedding_data = pickle.load(f)
            
            _self.professor_embeddings = embedding_data["embeddings"]
            return True, f"{len(_self.professor_embeddings)}개 임베딩 벡터 로드 완료"
        except FileNotFoundError:
            return False, "professor_embeddings.pkl 파일이 없습니다. 임베딩을 생성해주세요."
        except Exception as e:
            return False, f"임베딩 로드 실패: {str(e)}"
    
    def init_openai_client(self):
        """OpenAI 클라이언트 초기화"""
        try:
            api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
            api_version = os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or st.secrets.get("AZURE_OPENAI_ENDPOINT")
            
            if not api_key or not azure_endpoint:
                return False, "OpenAI API 키 또는 엔드포인트가 설정되지 않았습니다."
            
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint
            )
            return True, "OpenAI 클라이언트 초기화 완료"
        except Exception as e:
            return False, f"OpenAI 클라이언트 초기화 실패: {str(e)}"
    
    def create_professor_text_for_embedding(self, professor: Dict) -> str:
        """교수 정보를 임베딩용 텍스트로 변환"""
        parts = []
        
        # 기본정보
        name = professor["기본정보"]["교수이름"]
        degree = professor["기본정보"]["학위"]
        parts.append(f"교수명: {name}, 학위: {degree}")
        
        # 연구실
        if professor["연구실"]["연구실명"]:
            parts.append(f"연구실: {professor['연구실']['연구실명']}")
        
        # 연구분야 (가장 중요!)
        if professor["연구분야"]["키워드"]:
            parts.append(f"연구분야: {professor['연구분야']['키워드']}")
        
        if professor["연구분야"]["설명"]:
            description = professor["연구분야"]["설명"][:200]
            parts.append(f"연구설명: {description}")
        
        # 연구주제
        if professor["연구주제"]:
            topics = " | ".join(professor["연구주제"][:5])
            parts.append(f"연구주제: {topics}")
        
        # 기술방법
        if professor["기술및방법"]:
            methods = " | ".join(professor["기술및방법"][:5])
            parts.append(f"기술방법: {methods}")
        
        # 논문 (최신 3편만)
        if professor["논문"]:
            papers = " | ".join([paper[:100] for paper in professor["논문"][:3]])
            parts.append(f"최근논문: {papers}")
        
        return " / ".join(parts)
    
    def get_query_embedding(self, query: str) -> List[float]:
        """사용자 쿼리의 임베딩 벡터 생성"""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=query
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"임베딩 생성 실패: {e}")
            return [0.0] * 1536
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def find_similar_professors(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """쿼리와 유사한 교수들 찾기"""
        if not self.client:
            st.error("OpenAI 클라이언트가 초기화되지 않았습니다.")
            return []
        
        # 쿼리 임베딩
        query_embedding = self.get_query_embedding(query)
        
        # 모든 교수와 유사도 계산
        similarities = []
        for i, prof_embedding in enumerate(self.professor_embeddings):
            similarity = self.cosine_similarity(query_embedding, prof_embedding)
            similarities.append((self.professors_data[i], similarity))
        
        # 유사도 순으로 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def generate_recommendation_with_gpt(self, query: str, similar_professors: List[Tuple[Dict, float]]) -> str:
        """GPT-4o-mini로 최종 추천 생성"""
        if not self.client:
            return "OpenAI 클라이언트가 초기화되지 않았습니다."
        
        # 상위 매칭된 교수들만 GPT에게 전송
        top_professors = []
        for prof, similarity in similar_professors:
            prof_summary = {
                "이름": prof["기본정보"]["교수이름"],
                "연구실": prof["연구실"]["연구실명"],
                "키워드": prof["연구분야"]["키워드"][:150],  # 토큰 절약
                "연구주제": prof["연구주제"][:3],
                "기술방법": prof["기술및방법"][:3],
                "이메일": prof["기본정보"]["이메일"],
                "논문수": len(prof["논문"]),
                "유사도": f"{similarity:.3f}"
            }
            top_professors.append(prof_summary)
        
        prompt = f"""## 서울대학교 의과대학 연구실 추천

**학생 질문:** {query}

**벡터 임베딩으로 매칭된 상위 교수진:**
{json.dumps(top_professors, ensure_ascii=False, indent=2)}

**요청사항:**
위 학생의 질문을 바탕으로 가장 적합한 연구실을 순위별로 추천해주세요.

**답변 형식:**
### 🥇 1순위: [교수명] 교수 - [연구실명]
- **매칭도:** [유사도 점수] (매우 높음/높음/보통)
- **추천 이유:** [왜 이 연구실이 가장 적합한지 구체적으로 설명]
- **연구 분야:** [관련 키워드]
- **연락처:** [이메일]

### 🥈 2순위: [교수명] 교수 - [연구실명]
- **매칭도:** [유사도 점수]
- **추천 이유:** [구체적인 매칭 이유]
- **연구 분야:** [관련 키워드]
- **연락처:** [이메일]

### 🥉 3순위: [교수명] 교수 - [연구실명]
- **매칭도:** [유사도 점수]
- **추천 이유:** [구체적인 매칭 이유]
- **연구 분야:** [관련 키워드]
- **연락처:** [이메일]

**💡 추가 조언:** [해당 분야 연구를 위한 실용적인 조언]"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 서울대학교 의과대학 연구실 추천 전문가입니다. 벡터 임베딩으로 매칭된 결과를 바탕으로 학생에게 최적의 연구실을 추천해주세요. 추천 이유는 구체적이고 실용적으로 작성해주세요."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1200
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"추천 생성 중 오류 발생: {str(e)}"

def main():
    # 헤더
    st.title("🔬 서울대학교 의과대학 연구실 추천 시스템")
    st.markdown("**벡터 임베딩 + GPT-4o-mini 기반 맞춤형 연구실 추천**")
    
    # 시스템 초기화
    @st.cache_resource
    def init_system():
        return LabRecommendationSystem()
    
    recommender = init_system()
    
    # 사이드바 - 시스템 상태
    with st.sidebar:
        st.header("🔧 시스템 상태")
        
        # 데이터 로드 상태
        data_success, data_msg = recommender.load_professor_data()
        if data_success:
            st.success(data_msg)
        else:
            st.error(data_msg)
            st.stop()
        
        # 임베딩 로드 상태  
        embed_success, embed_msg = recommender.load_embeddings()
        if embed_success:
            st.success(embed_msg)
        else:
            st.warning(embed_msg)
            st.info("시뮬레이션 모드로 실행됩니다.")
        
        # OpenAI 클라이언트 상태
        if embed_success:  # 임베딩이 있을 때만 OpenAI 초기화
            openai_success, openai_msg = recommender.init_openai_client()
            if openai_success:
                st.success(openai_msg)
            else:
                st.error(openai_msg)
                st.info("환경변수 또는 Streamlit secrets에 API 키를 설정해주세요.")
        
        st.markdown("---")
        st.markdown("### 📊 시스템 정보")
        st.markdown(f"- **교수 수**: {len(recommender.professors_data)}명")
        st.markdown(f"- **임베딩 모델**: text-embedding-3-small")
        st.markdown(f"- **추천 모델**: GPT-4o-mini")
    
    # 메인 컨텐츠
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.header("💬 연구 관심분야 입력")
        
        # 예시 쿼리 버튼들
        st.markdown("**빠른 예시:**")
        example_queries = [
            "암 치료와 관련된 나노기술 연구에 관심이 있습니다",
            "뇌과학과 인공지능을 결합한 연구를 하고 싶어요",
            "면역학과 세포생물학 분야에서 연구하고 싶습니다",
            "영상의학과 진단기술 개발에 관심있어요"
        ]
        
        selected_example = None
        cols = st.columns(2)
        for i, example in enumerate(example_queries):
            with cols[i % 2]:
                if st.button(f"예시 {i+1}", key=f"example_{i}"):
                    selected_example = example
        
        # 사용자 입력
        user_query = st.text_area(
            "관심있는 연구분야를 자세히 설명해주세요:",
            value=selected_example if selected_example else "",
            height=150,
            placeholder="예: 암 진단을 위한 나노입자 기반 영상 기술에 관심이 있습니다. 특히 PET/CT를 이용한 분자영상 분야에서 연구하고 싶어요."
        )
        
        # 추천 옵션
        st.markdown("### ⚙️ 추천 옵션")
        top_k = st.slider("추천받을 연구실 수", min_value=3, max_value=7, value=5)
        
        # 추천 실행 버튼
        recommend_button = st.button("🎯 연구실 추천받기", type="primary", use_container_width=True)
    
    with col2:
        st.header("📋 추천 결과")
        
        if recommend_button and user_query.strip():
            if not embed_success:
                st.error("임베딩 데이터가 없어 추천을 실행할 수 없습니다.")
            else:
                with st.spinner("🔍 벡터 유사도 계산 중..."):
                    # 벡터 매칭
                    similar_professors = recommender.find_similar_professors(user_query, top_k)
                    
                    if similar_professors:
                        # 매칭 결과 미리보기
                        st.markdown("### 🎯 벡터 매칭 결과")
                        match_df_data = []
                        for i, (prof, similarity) in enumerate(similar_professors, 1):
                            match_df_data.append({
                                "순위": i,
                                "교수명": prof["기본정보"]["교수이름"],
                                "유사도": f"{similarity:.3f}",
                                "연구분야": prof["연구분야"]["키워드"][:50] + "..." if len(prof["연구분야"]["키워드"]) > 50 else prof["연구분야"]["키워드"]
                            })
                        
                        st.dataframe(match_df_data, use_container_width=True)
                        
                        # GPT 추천 생성
                        if recommender.client:
                            with st.spinner("🤖 GPT-4o-mini가 상세 추천 생성 중..."):
                                recommendation = recommender.generate_recommendation_with_gpt(user_query, similar_professors)
                                
                                st.markdown("### 🎓 AI 추천 결과")
                                st.markdown(recommendation)
                        else:
                            st.warning("OpenAI API가 설정되지 않아 벡터 매칭 결과만 표시됩니다.")
                    else:
                        st.error("매칭된 연구실을 찾을 수 없습니다.")
        
        elif recommend_button:
            st.warning("연구 관심분야를 입력해주세요.")
        
        else:
            st.info("👈 왼쪽에서 관심있는 연구분야를 입력하고 '연구실 추천받기' 버튼을 클릭해주세요.")
            
            # 시스템 소개
            st.markdown("### 🚀 시스템 특징")
            st.markdown("""
            - **벡터 임베딩**: text-embedding-3-small으로 의미론적 유사도 계산
            - **AI 추천**: GPT-4o-mini가 구체적인 추천 이유 제공  
            - **실시간 매칭**: 31명 교수진 중 최적 매칭
            - **완전한 정보**: 연구분야, 주제, 기술, 최신 논문 포함
            """)

if __name__ == "__main__":
    main()