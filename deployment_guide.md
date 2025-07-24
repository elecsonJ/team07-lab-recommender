# 🌐 대학원 연구실 추천 AI 배포 가이드

## 📋 배포 옵션별 비교

| 옵션 | 비용 | 난이도 | 확장성 | 추천도 |
|------|------|--------|--------|---------|
| **Streamlit Cloud** | 무료 | ⭐ | ⭐⭐ | 🏆 **추천** |
| **Heroku** | $7/월 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **AWS EC2** | $10-30/월 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Google Cloud Run** | 사용량 기반 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Railway** | $5/월 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

---

## 🏆 **1. Streamlit Cloud (가장 추천)**

### ✅ **장점**
- **완전 무료** (공개 레포용)
- Streamlit 최적화
- 자동 SSL 인증서
- GitHub 연동으로 자동 배포
- 설정 최소화

### 📝 **배포 단계**

#### 1단계: GitHub 레포지토리 준비
```bash
# 새 레포지토리 생성
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/team07-lab-recommender.git
git push -u origin main
```

#### 2단계: 환경변수 설정 파일 생성
```bash
# .streamlit/secrets.toml 파일 생성 (GitHub에는 업로드하지 않음)
cat > .streamlit/secrets.toml << 'EOF'
AZURE_OPENAI_ENDPOINT = "your_endpoint_here"
OPENAI_API_KEY = "your_api_key_here"
OPENAI_API_VERSION = "your_version_here"
EOF

# .gitignore에 추가
echo ".streamlit/secrets.toml" >> .gitignore
```

#### 3단계: Streamlit Cloud 배포
1. https://share.streamlit.io 접속
2. GitHub 연동
3. 레포지토리 선택: `team07-lab-recommender`
4. 메인 파일: `streamlit_app.py`
5. **Secrets** 탭에서 환경변수 입력:
   ```
   AZURE_OPENAI_ENDPOINT = "your_endpoint_here"
   OPENAI_API_KEY = "your_api_key_here"  
   OPENAI_API_VERSION = "your_version_here"
   ```
6. 배포 시작

#### 4단계: 접속 URL
- 자동 생성: `https://YOUR_USERNAME-team07-lab-recommender-streamlit-app-xxxxx.streamlit.app`

---

## 🐳 **2. Docker + 클라우드 배포**

### Dockerfile 생성
```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# 앱 파일 복사
COPY . .

# 포트 노출
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 앱 실행
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Compose (로컬 테스트용)
```yaml
# docker-compose.yml
version: '3.8'
services:
  streamlit-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_API_VERSION=${OPENAI_API_VERSION}
    volumes:
      - ./chroma_db:/app/chroma_db
```

---

## ☁️ **3. Google Cloud Run 배포**

### 배포 스크립트
```bash
#!/bin/bash
# deploy.sh

# 프로젝트 설정
PROJECT_ID="your-project-id"
SERVICE_NAME="lab-recommender"
REGION="asia-northeast3"

# Docker 이미지 빌드 및 푸시
gcloud builds submit --tag gcr.io/${PROJECT_ID}/${SERVICE_NAME}

# Cloud Run 배포
gcloud run deploy ${SERVICE_NAME} \
    --image gcr.io/${PROJECT_ID}/${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --port 8501 \
    --set-env-vars AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT} \
    --set-env-vars OPENAI_API_KEY=${OPENAI_API_KEY} \
    --set-env-vars OPENAI_API_VERSION=${OPENAI_API_VERSION}
```

---

## 🚂 **4. Railway 배포**

### 간단한 배포
1. https://railway.app 접속
2. GitHub 연동
3. 레포지토리 선택
4. 환경변수 설정
5. 자동 배포

### railway.json 설정
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "startCommand": "streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0",
    "healthcheckPath": "/_stcore/health"
  }
}
```

---

## 🔧 **5. 추가 최적화 (선택사항)**

### 성능 최적화
```python
# streamlit_config.py
import streamlit as st

# 캐싱 설정
@st.cache_data(ttl=3600)  # 1시간 캐시
def load_rag_system():
    return LabRecommenderRAG(data_path)

@st.cache_resource
def get_vector_store():
    # 벡터 저장소 캐싱
    pass
```

### CDN 및 정적 파일 최적화
```python
# requirements.txt에 추가
streamlit-aggrid==0.3.4
streamlit-chat==0.1.1
streamlit-option-menu==0.3.6
```

---

## 🛡️ **6. 보안 고려사항**

### API 키 보호
- ✅ 환경변수 사용
- ✅ secrets.toml 파일 `.gitignore`에 추가
- ✅ 클라우드 시크릿 매니저 활용

### 사용량 제한
```python
# 요청 제한 (선택사항)
import time
from functools import wraps

def rate_limit(calls_per_minute=10):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 간단한 속도 제한 로직
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

---

## 🎯 **최종 추천: Streamlit Cloud**

**왜 Streamlit Cloud인가?**
1. **무료** - 개인/학습 프로젝트에 최적
2. **간단함** - 클릭 몇 번으로 배포 완료
3. **안정성** - Streamlit 공식 플랫폼
4. **자동화** - GitHub 푸시시 자동 재배포
5. **SSL** - HTTPS 자동 제공

**배포 후 확인사항:**
- [ ] 앱 정상 로드 확인
- [ ] 채팅 기능 테스트
- [ ] 질문 분류 동작 확인
- [ ] 연구실 추천 결과 확인
- [ ] 모바일 반응형 확인

**접속 URL 예시:**
```
https://your-app-name.streamlit.app
```

이제 전 세계 누구나 웹 브라우저로 대학원 연구실 추천 AI를 사용할 수 있습니다! 🌍