# 🎓 대학원 연구실 추천 AI

RAG(Retrieval-Augmented Generation) 기술을 활용한 대학원 연구실 추천 시스템

## 🚀 웹 앱 접속

배포된 웹 앱: [https://team07-lab-recommender.streamlit.app](https://team07-lab-recommender.streamlit.app)

## 📋 주요 기능

- **지능형 질문 분류**: 사용자 질문을 자동으로 분석하여 적절한 검색 방식 선택
- **맞춤형 연구실 추천**: 관심 분야에 맞는 교수/연구실 추천
- **대화형 인터페이스**: 자연스러운 대화를 통한 상세 질문 및 답변
- **실시간 응답**: GPT-4o-mini 모델 기반 빠른 응답

## 🔧 기술 스택

- **Frontend**: Streamlit
- **Backend**: LangChain + GPT-4o-mini
- **Vector DB**: Chroma
- **Embeddings**: text-embedding-3-small
- **Data**: 서울대학교 의과대학 교수진 31명

## 💡 사용법

1. 웹 앱에 접속
2. 관심 있는 연구 분야 입력 (예: "인공지능 연구하고 싶어")
3. AI가 적합한 연구실 추천
4. 추가 질문으로 세부 정보 확인

## 🏗️ 로컬 실행

```bash
# 1. 레포지토리 클론
git clone https://github.com/YOUR_USERNAME/team07-lab-recommender.git
cd team07-lab-recommender

# 2. 가상환경 생성 및 활성화
python -m venv team07_env
source team07_env/bin/activate  # Windows: team07_env\Scripts\activate

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 환경변수 설정 (.env 파일 생성)
AZURE_OPENAI_ENDPOINT=your_endpoint
OPENAI_API_KEY=your_api_key
OPENAI_API_VERSION=your_version

# 5. 웹 앱 실행
streamlit run streamlit_app.py
```

## 📊 데이터 현황

- **교수 수**: 31명
- **논문 수**: 96편  
- **데이터 완성도**: 95.8%
- **마지막 업데이트**: 2025-07-24

## 🤝 팀원

Team 07 - 대학원 연구실 추천 시스템 개발팀