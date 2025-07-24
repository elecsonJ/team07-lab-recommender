"""
교수진 벡터 임베딩 생성 스크립트
"""

import json
import os
import pickle
from openai import AzureOpenAI
from typing import List, Dict, Any

class EmbeddingGenerator:
    def __init__(self):
        self.professors_data = []
        self.professor_embeddings = []
        self.embedding_model = "text-embedding-3-small"
        
        # Azure OpenAI 클라이언트 초기화
        self.client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    
    def load_professor_data(self, json_path: str):
        """교수진 데이터 로드"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.professors_data = data["교수진"]
        print(f"✅ {len(self.professors_data)}명 교수 데이터 로드 완료")
        return self.professors_data
    
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
    
    def generate_all_embeddings(self):
        """모든 교수에 대한 임베딩 벡터 생성"""
        print("🔄 교수별 임베딩 벡터 생성 중...")
        
        self.professor_embeddings = []
        
        for i, professor in enumerate(self.professors_data):
            # 임베딩용 텍스트 생성
            text = self.create_professor_text_for_embedding(professor)
            
            try:
                # Azure OpenAI 임베딩 API 호출
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                
                embedding = response.data[0].embedding
                self.professor_embeddings.append(embedding)
                
                prof_name = professor["기본정보"]["교수이름"]
                print(f"  ✅ {i+1}/31: {prof_name} 임베딩 완료")
                
            except Exception as e:
                print(f"  ❌ {professor['기본정보']['교수이름']} 임베딩 실패: {e}")
                # 실패한 경우 0 벡터로 채움
                self.professor_embeddings.append([0.0] * 1536)
        
        print(f"📊 총 {len(self.professor_embeddings)}개 임베딩 벡터 생성 완료")
        return self.professor_embeddings
    
    def save_embeddings(self, filepath: str = "professor_embeddings.pkl"):
        """임베딩 벡터 저장"""
        embedding_data = {
            "embeddings": self.professor_embeddings,
            "professor_names": [prof["기본정보"]["교수이름"] for prof in self.professors_data],
            "model": self.embedding_model,
            "total_count": len(self.professor_embeddings),
            "dimension": 1536
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(embedding_data, f)
        
        print(f"💾 임베딩 벡터 저장 완료: {filepath}")
        print(f"📊 {len(self.professor_embeddings)}개 벡터, 1536차원")

def main():
    print("🚀 교수진 벡터 임베딩 생성 시작")
    print("="*50)
    
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("다음 명령어로 설정해주세요:")
        print("export OPENAI_API_KEY='your_api_key'")
        return
    
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("❌ AZURE_OPENAI_ENDPOINT 환경변수가 설정되지 않았습니다.")
        return
    
    generator = EmbeddingGenerator()
    
    try:
        # 1. 교수 데이터 로드
        generator.load_professor_data("professors_final_complete.json")
        
        # 2. 임베딩 생성
        generator.generate_all_embeddings()
        
        # 3. 임베딩 저장
        generator.save_embeddings()
        
        print("\n🎉 임베딩 생성 완료!")
        print("이제 streamlit run streamlit_lab_recommender.py 로 웹앱을 실행할 수 있습니다.")
        
    except FileNotFoundError:
        print("❌ professors_final_complete.json 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()