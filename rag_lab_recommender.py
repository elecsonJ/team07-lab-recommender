import os
import json
import numpy as np
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import argparse
from typing import Dict, List, Any
from dataclasses import dataclass, field

# 환경변수 로드
load_dotenv()

@dataclass
class ConversationHistory:
    """대화 히스토리 관리 클래스"""
    queries: List[str] = field(default_factory=list)
    responses: List[str] = field(default_factory=list)
    retrieved_docs: List[List[Document]] = field(default_factory=list)
    
    def add_turn(self, query: str, response: str, docs: List[Document] = None):
        self.queries.append(query)
        self.responses.append(response)
        if docs:
            self.retrieved_docs.append(docs)
    
    def get_context(self, last_n: int = 3) -> str:
        """최근 n개의 대화 컨텍스트 반환"""
        context = []
        start_idx = max(0, len(self.queries) - last_n)
        
        for i in range(start_idx, len(self.queries)):
            context.append(f"Q: {self.queries[i]}")
            if i < len(self.responses):
                context.append(f"A: {self.responses[i][:200]}...")  # 응답은 200자로 제한
        
        return "\n".join(context)
    
    def clear(self):
        """히스토리 초기화"""
        self.queries.clear()
        self.responses.clear()
        self.retrieved_docs.clear()

class LabRecommenderRAG:
    def __init__(self, data_path, vector_store_path="./vector_store"):
        self.data_path = data_path
        self.vector_store_path = vector_store_path
        
        # Azure OpenAI 임베딩 모델 초기화
        self.embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-3-small",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION"),
            dimensions=1536
        )
        
        # Azure OpenAI LLM 모델 초기화
        self.llm = AzureChatOpenAI(
            model="gpt-4o-mini",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION"),
            temperature=0.3
        )
        
        self.vector_store = None
        self.qa_chain = None
        self.conversation_history = ConversationHistory()
        
    def load_and_process_data(self):
        """교수 데이터를 로드하고 Document 객체로 변환"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for professor in data['교수진']:
            # 전체 교수 정보를 포함한 상세 텍스트 생성
            prof_info = f"""
=== 기본 정보 ===
교수명: {professor['기본정보']['교수이름']}
대학명: {professor['기본정보'].get('대학명', '')}
학과명: {professor['기본정보'].get('학과명', '')}
연구실: {professor['연구실']['연구실명']}
연구분야: {professor['연구분야']['키워드']}
연구분야 설명: {professor['연구분야']['설명']}
이메일: {professor['기본정보']['이메일']}
전화번호: {professor['기본정보']['전화번호']}
학위: {professor['기본정보']['학위']}
연구실 웹사이트: {professor['연구실'].get('연구실웹사이트', '')}

=== 연구 상세 정보 ===
연구주제:
{chr(10).join(professor['연구주제']) if professor['연구주제'] else '정보 없음'}

기술 및 방법:
{chr(10).join(professor['기술및방법']) if professor['기술및방법'] else '정보 없음'}

학력 및 경력:
{chr(10).join(professor['학력경력']) if professor['학력경력'] else '정보 없음'}

주요 논문:
{chr(10).join(professor['논문']) if professor['논문'] else '정보 없음'}

학생지도 특징:
{professor['학생지도'].get('특징', '정보 없음')}

학생 진로:
{professor['학생지도'].get('진로', '정보 없음')}
            """.strip()
            
            # 메타데이터 설정
            metadata = {
                "professor_name": professor['기본정보']['교수이름'],
                "university": professor['기본정보'].get('대학명', ''),
                "department": professor['기본정보'].get('학과명', ''),
                "lab_name": professor['연구실']['연구실명'],
                "email": professor['기본정보']['이메일'],
                "phone": professor['기본정보']['전화번호'],
                "keywords": professor['연구분야']['키워드']
            }
            
            documents.append(Document(
                page_content=prof_info,
                metadata=metadata
            ))
        
        return documents
    
    def create_vector_store(self):
        """벡터 저장소 생성"""
        print("교수 데이터를 로드하고 있습니다...")
        documents = self.load_and_process_data()
        
        print("벡터 임베딩을 생성하고 있습니다...")
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        # FAISS 인덱스 저장
        self.vector_store.save_local(self.vector_store_path)
        print(f"벡터 저장소가 {self.vector_store_path}에 저장되었습니다.")
    
    def load_vector_store(self):
        """기존 벡터 저장소 로드"""
        try:
            self.vector_store = FAISS.load_local(
                self.vector_store_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("기존 벡터 저장소를 로드했습니다.")
            return True
        except Exception as e:
            print(f"벡터 저장소 로드 실패: {e}")
            return False
    
    def contains_professor_name(self, query: str) -> bool:
        """질문에 교수명이 포함되어 있는지 확인"""
        professor_names = [
            "강건욱", "강홍기", "구자록", "김동현", "김명환", "김성준", "김종일", "김현제", "김현진",
            "박상민", "박성준", "박수경", "박정규", "방예지", "서인석", "성지혜", "신현무", "여선주",
            "이민재", "이용석", "이지연", "이진구", "이창한", "이철환", "조성엽", "조주연", "최경호",
            "최민호", "최은영", "최형진", "신정환"
        ]
        for name in professor_names:
            if name in query:
                return True
        return False
    
    def can_answer_with_previous(self, query: str) -> bool:
        """이전 검색 결과로 답변 가능한지 확인"""
        if not self.conversation_history.retrieved_docs:
            return False
        
        # 단순한 후속 질문 패턴 확인
        followup_patterns = ["그 중에서", "더 자세히", "추가로", "그런데", "또", "그리고"]
        return any(pattern in query for pattern in followup_patterns)
    
    def is_research_related(self, query: str) -> bool:
        """연구분야 관련 질문인지 확인"""
        research_keywords = [
            "연구", "AI", "인공지능", "머신러닝", "바이오", "의료", "생명과학", "분자", "세포",
            "유전", "면역", "영상", "신경", "뇌", "암", "종양", "치료", "진단", "약물",
            "실험실", "연구실", "교수", "추천", "관심", "하고싶다", "배우고싶다"
        ]
        return any(keyword in query for keyword in research_keywords)
    
    def classify_query(self, new_query: str) -> Dict[str, Any]:
        """개선된 질문 분류 시스템"""
        # 1. 교수명 언급 체크
        if self.contains_professor_name(new_query):
            return {"type": "professor_detail", "reason": "특정 교수 언급"}
        
        # 2. 이전 결과로 답변 가능한지 체크
        if self.can_answer_with_previous(new_query):
            return {"type": "refine_previous", "reason": "이전 결과 활용 가능"}
        
        # 3. 연구분야 관련 질문인지 체크
        if self.is_research_related(new_query):
            return {"type": "new_search", "reason": "새로운 연구분야 검색"}
        
        # 4. 나머지는 일반 질문
        return {"type": "general_info", "reason": "대학원 일반 정보"}
    
    def setup_qa_chain(self, k=5):
        """RAG QA 체인 설정"""
        if self.vector_store is None:
            raise ValueError("벡터 저장소가 초기화되지 않았습니다.")
        
        # MMR 검색기 설정 (Maximum Marginal Relevance)
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": k*2,  # MMR을 위해 더 많은 후보를 가져옴
                "lambda_mult": 0.5  # 다양성과 관련성의 균형 조절 (0~1)
            }
        )
        
        # 프롬프트 템플릿 정의
        prompt_template = """다음은 대학원 교수진의 상세 정보입니다. 학생의 질문에 기반하여 적절한 답변을 제공해주세요.

교수진 정보:
{context}

학생의 질문: {question}

답변 가이드라인:

**초기 연구실 추천의 경우:**
1. 학생의 관심 분야와 가장 유사한 연구 분야를 가진 교수를 2-3명 추천
2. 각 교수에 대해 간략하게 다음 정보만 제시:
   - 교수명과 연구실명
   - 대학명과 학과명 (서울대학교 의과대학)
   - 핵심 연구분야 (1-2줄 요약)
   - 추천 이유 (1-2줄)
3. "더 자세한 정보가 궁금하시면 특정 교수님에 대해 질문해주세요!"라고 안내

**특정 교수에 대한 세부 질문의 경우:**
1. 해당 교수의 상세 정보를 활용하여 구체적으로 답변
2. 논문 제목 해석, 연구 트렌드 분석, 학력 배경 등 심화 정보 제공
3. 질문 유형에 따라 적절한 정보 선별 (논문/연구주제/학력 등)

**일반 질문의 경우:**
1. 대학원 생활, 입학 절차 등에 대해 일반적인 조언 제공

한국어로 친근하고 도움이 되는 톤으로 답변해주세요.

답변:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # QA 체인 생성
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def get_recommendation(self, user_query):
        """사용자 질문에 대한 연구실 추천 (구버전 호환용)"""
        if self.qa_chain is None:
            raise ValueError("QA 체인이 설정되지 않았습니다.")
        
        print("\n🔍 관련 연구실을 검색하고 있습니다...")
        
        result = self.qa_chain.invoke({"query": user_query})
        
        print("\n" + "="*60)
        print("🎯 연구실 추천 결과")
        print("="*60)
        print(result["result"])
        
        return result
    
    def process_new_search(self, user_query: str) -> Dict[str, Any]:
        """새로운 검색 처리 - 간략한 추천 모드"""
        print("\n🔍 새로운 검색을 시작합니다...")
        result = self.qa_chain.invoke({"query": user_query})
        return result
    
    def process_refine_previous(self, user_query: str) -> Dict[str, Any]:
        """이전 결과 내에서 재검색"""
        if not self.conversation_history.retrieved_docs:
            # 이전 결과가 없으면 새 검색
            return self.process_new_search(user_query)
        
        print("\n🔄 이전 추천 결과를 바탕으로 답변합니다...")
        
        # 이전 추천 교수 정보만 컨텍스트로 사용
        previous_docs = self.conversation_history.retrieved_docs[-1]
        context_text = "\n\n".join([doc.page_content for doc in previous_docs[:5]])
        
        refined_prompt = f"""
이전에 추천한 교수진 정보:
{context_text}

학생의 추가 질문: {user_query}

위 교수진 정보를 바탕으로 추가 질문에 답변해주세요.
"""
        
        response = self.llm.invoke(refined_prompt)
        return {"result": response.content, "source_documents": previous_docs}
    
    def process_professor_detail(self, user_query: str) -> Dict[str, Any]:
        """특정 교수 상세 정보 처리"""
        print("\n👨‍🏫 특정 교수님에 대한 상세 정보를 검색합니다...")
        result = self.qa_chain.invoke({"query": user_query})
        return result
    
    def process_general_info(self, user_query: str) -> Dict[str, Any]:
        """일반 정보 처리 (RAG 없이)"""
        print("\n💬 대학원 일반 정보에 답변합니다...")
        
        general_prompt = f"""
대학원 일반 질문: {user_query}

다음과 같은 주제에 대해 도움을 드릴 수 있습니다:
- 대학원 입학 절차 및 준비사항
- 연구실 생활 및 연구 과정
- 지원 자격 및 요구사항
- 대학원생으로서의 일반적인 조언

친근하고 도움이 되는 톤으로 답변해주세요.
"""
        
        response = self.llm.invoke(general_prompt)
        return {"result": response.content, "source_documents": []}
    
    def process_query(self, user_query: str) -> str:
        """질문 분류 후 적절한 처리"""
        # 질문 분류
        classification = self.classify_query(user_query)
        query_type = classification.get("type", "new_search")
        reason = classification.get("reason", "")
        
        print(f"\n🤖 질문 분류: {query_type}")
        print(f"   이유: {reason}")
        
        # 분류에 따른 처리
        if query_type == "new_search":
            result = self.process_new_search(user_query)
        elif query_type == "refine_previous":
            result = self.process_refine_previous(user_query)
        elif query_type == "professor_detail":
            result = self.process_professor_detail(user_query)
        elif query_type == "general_info":
            result = self.process_general_info(user_query)
        else:
            result = self.process_new_search(user_query)
        
        # 히스토리에 저장
        response_text = result["result"]
        source_docs = result.get("source_documents", [])
        self.conversation_history.add_turn(user_query, response_text, source_docs)
        
        return response_text

def main():
    parser = argparse.ArgumentParser(description='대학원 연구실 추천 AI')
    parser.add_argument('--rebuild', action='store_true', 
                       help='벡터 저장소를 새로 생성합니다')
    parser.add_argument('--k', type=int, default=5,
                       help='검색할 연구실 수 (기본값: 5)')
    
    args = parser.parse_args()
    
    # 데이터 경로
    data_path = "professors_final_complete.json"
    
    # RAG 시스템 초기화
    rag_system = LabRecommenderRAG(data_path)
    
    # 벡터 저장소 설정
    if args.rebuild or not rag_system.load_vector_store():
        rag_system.create_vector_store()
    
    # QA 체인 설정
    rag_system.setup_qa_chain(k=args.k)
    
    print("\n🎓 대학원 연구실 추천 AI에 오신 것을 환영합니다!")
    print("관심있는 연구 분야나 주제를 자유롭게 입력해주세요.")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print("대화를 새로 시작하려면 'clear' 또는 'reset'을 입력하세요.\n")
    
    # 첫 질문 여부 추적
    is_first_question = True
    
    while True:
        try:
            # 대화형 프롬프트 개선
            if is_first_question:
                prompt = "\n💭 어떤 연구 분야에 관심이 있으신가요? >> "
            else:
                prompt = "\n💬 추가 질문이 있으신가요? >> "
            
            user_input = input(prompt).strip()
            
            if user_input.lower() in ['quit', 'exit', '종료', '끝']:
                print("\n👋 대학원 연구실 추천 AI를 이용해 주셔서 감사합니다!")
                break
            
            if user_input.lower() in ['clear', 'reset', '초기화', '새로시작']:
                rag_system.conversation_history.clear()
                print("\n🔄 대화 히스토리가 초기화되었습니다. 새로운 대화를 시작합니다.")
                is_first_question = True
                continue
            
            if not user_input:
                print("❗ 질문을 입력해주세요.")
                continue
            
            # 질문 처리 및 응답
            print("\n" + "="*60)
            response = rag_system.process_query(user_input)
            print("="*60)
            print(response)
            print("="*60)
            
            is_first_question = False
            
        except EOFError:
            print("\n\n👋 프로그램을 종료합니다.")
            break
        except KeyboardInterrupt:
            print("\n\n👋 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류가 발생했습니다: {e}")
            print("다시 시도해주세요.")
            continue

if __name__ == "__main__":
    main()