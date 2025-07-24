"""
Playwright를 이용한 Streamlit UI 자동 테스트
"""
import asyncio
import subprocess
import time
import signal
import os
from pathlib import Path

def start_streamlit_server():
    """Streamlit 서버 시작"""
    script_dir = Path(__file__).parent
    streamlit_app = script_dir / "streamlit_app.py"
    
    # Streamlit 서버 시작
    process = subprocess.Popen([
        "streamlit", "run", str(streamlit_app),
        "--server.port=8502",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 서버가 준비될 때까지 대기
    print("🔄 Streamlit 서버 시작 중...")
    time.sleep(10)  # 서버 초기화 대기
    
    return process

def stop_streamlit_server(process):
    """Streamlit 서버 종료"""
    if process:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("🛑 Streamlit 서버 종료됨")

async def run_ui_tests():
    """UI 테스트 실행"""
    from playwright.async_api import async_playwright
    
    streamlit_process = None
    
    try:
        # Streamlit 서버 시작
        streamlit_process = start_streamlit_server()
        
        async with async_playwright() as p:
            # 브라우저 실행
            browser = await p.chromium.launch(headless=False, slow_mo=1000)
            page = await browser.new_page()
            
            # Streamlit 앱으로 이동
            print("🌐 Streamlit 앱 접속 중...")
            await page.goto("http://localhost:8502")
            
            # 페이지 로드 대기
            await page.wait_for_load_state("networkidle")
            
            # 테스트 시작
            print("✅ 테스트 시작")
            
            # 1. 페이지 제목 확인
            await test_page_title(page)
            
            # 2. 초기 UI 요소 확인
            await test_initial_ui_elements(page)
            
            # 3. 사이드바 기능 테스트
            await test_sidebar_functionality(page)
            
            # 4. 챗봇 기본 대화 테스트
            await test_basic_chat_functionality(page)
            
            # 5. 질문 분류 테스트
            await test_question_classification(page)
            
            # 6. 대화 히스토리 테스트
            await test_conversation_history(page)
            
            # 7. 특수 명령어 테스트
            await test_special_commands(page)
            
            # 8. 에러 처리 테스트
            await test_error_handling(page)
            
            print("🎉 모든 테스트 완료!")
            
            # 브라우저 종료
            await browser.close()
    
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")
        raise
    
    finally:
        # Streamlit 서버 종료
        if streamlit_process:
            stop_streamlit_server(streamlit_process)

async def test_page_title(page):
    """페이지 제목 테스트"""
    print("🔍 테스트 1: 페이지 제목 확인")
    
    title = await page.title()
    assert "대학원 연구실 추천 AI" in title, f"페이지 제목이 올바르지 않음: {title}"
    
    # 메인 헤더 확인
    header = await page.locator(".main-header").text_content()
    assert "대학원 연구실 추천 AI" in header, "메인 헤더가 올바르지 않음"
    
    print("✅ 페이지 제목 테스트 통과")

async def test_initial_ui_elements(page):
    """초기 UI 요소 테스트"""
    print("🔍 테스트 2: 초기 UI 요소 확인")
    
    # 사이드바 확인
    sidebar = page.locator(".css-1d391kg")  # Streamlit 사이드바 클래스
    assert await sidebar.is_visible(), "사이드바가 보이지 않음"
    
    # 채팅 입력창 확인
    chat_input = page.locator("[data-testid='stChatInputTextArea']")
    assert await chat_input.is_visible(), "채팅 입력창이 보이지 않음"
    
    # 안내 메시지 확인
    info_message = page.locator(".stAlert")
    assert await info_message.is_visible(), "안내 메시지가 보이지 않음"
    
    print("✅ 초기 UI 요소 테스트 통과")

async def test_sidebar_functionality(page):
    """사이드바 기능 테스트"""
    print("🔍 테스트 3: 사이드바 기능 확인")
    
    # 대화 초기화 버튼 확인
    reset_button = page.locator("text=🔄 대화 초기화")
    assert await reset_button.is_visible(), "대화 초기화 버튼이 보이지 않음"
    
    # 벡터 저장소 재구축 버튼 확인
    rebuild_button = page.locator("text=🔧 벡터 저장소 재구축")
    assert await rebuild_button.is_visible(), "벡터 저장소 재구축 버튼이 보이지 않음"
    
    # 사용법 안내 확인
    usage_guide = page.locator("text=📖 사용법")
    assert await usage_guide.is_visible(), "사용법 안내가 보이지 않음"
    
    print("✅ 사이드바 기능 테스트 통과")

async def test_basic_chat_functionality(page):
    """기본 챗봇 기능 테스트"""
    print("🔍 테스트 4: 기본 챗봇 기능 확인")
    
    # 채팅 입력창에 메시지 입력
    chat_input = page.locator("[data-testid='stChatInputTextArea']")
    await chat_input.fill("안녕하세요")
    await chat_input.press("Enter")
    
    # 응답 대기 (최대 30초)
    await page.wait_for_timeout(5000)
    
    # 사용자 메시지 확인
    user_messages = page.locator(".chat-message.user")
    assert await user_messages.count() > 0, "사용자 메시지가 표시되지 않음"
    
    # AI 응답 확인
    assistant_messages = page.locator(".chat-message.assistant")
    assert await assistant_messages.count() > 0, "AI 응답이 표시되지 않음"
    
    print("✅ 기본 챗봇 기능 테스트 통과")

async def test_question_classification(page):
    """질문 분류 테스트"""
    print("🔍 테스트 5: 질문 분류 기능 확인")
    
    # AI 연구 관련 질문 (new_search 예상)
    chat_input = page.locator("[data-testid='stChatInputTextArea']")
    await chat_input.fill("인공지능 연구하고 싶어")
    await chat_input.press("Enter")
    
    # 응답 대기
    await page.wait_for_timeout(10000)
    
    # 분류 정보 확인
    classification_info = page.locator(".classification-info")
    assert await classification_info.count() > 0, "질문 분류 정보가 표시되지 않음"
    
    # 분류 유형 확인
    classification_text = await classification_info.last.text_content()
    assert "new_search" in classification_text, f"예상과 다른 분류: {classification_text}"
    
    print("✅ 질문 분류 테스트 통과")

async def test_conversation_history(page):
    """대화 히스토리 테스트"""
    print("🔍 테스트 6: 대화 히스토리 확인")
    
    # 추가 질문 (refine_previous 예상)
    chat_input = page.locator("[data-testid='stChatInputTextArea']")
    await chat_input.fill("그 중에서 의료 AI는?")
    await chat_input.press("Enter")
    
    # 응답 대기
    await page.wait_for_timeout(10000)
    
    # 메시지 수 확인 (최소 4개: 안녕 + 응답 + AI연구 + 응답 + 의료AI + 응답)
    all_messages = page.locator(".chat-message")
    message_count = await all_messages.count()
    assert message_count >= 6, f"대화 히스토리가 올바르지 않음: {message_count}개 메시지"
    
    # 분류가 refine_previous인지 확인
    last_classification = page.locator(".classification-info").last
    classification_text = await last_classification.text_content()
    assert "refine_previous" in classification_text, f"잘못된 분류: {classification_text}"
    
    print("✅ 대화 히스토리 테스트 통과")

async def test_special_commands(page):
    """특수 명령어 테스트"""
    print("🔍 테스트 7: 특수 명령어 확인")
    
    # 현재 메시지 수 확인
    initial_messages = await page.locator(".chat-message").count()
    
    # clear 명령어 테스트
    chat_input = page.locator("[data-testid='stChatInputTextArea']")
    await chat_input.fill("clear")
    await chat_input.press("Enter")
    
    # 페이지 새로고침 대기
    await page.wait_for_timeout(3000)
    
    # 메시지가 초기화되었는지 확인
    final_messages = await page.locator(".chat-message").count()
    assert final_messages == 0, f"대화 초기화 실패: {final_messages}개 메시지 남음"
    
    print("✅ 특수 명령어 테스트 통과")

async def test_error_handling(page):
    """에러 처리 테스트"""
    print("🔍 테스트 8: 에러 처리 확인")
    
    # 매우 긴 입력 테스트
    long_input = "a" * 1000
    chat_input = page.locator("[data-testid='stChatInputTextArea']")
    await chat_input.fill(long_input)
    await chat_input.press("Enter")
    
    # 응답 대기
    await page.wait_for_timeout(5000)
    
    # 에러 메시지나 정상 응답 확인
    error_elements = page.locator(".stAlert")
    chat_messages = page.locator(".chat-message")
    
    # 에러가 발생했거나 정상 응답이 있어야 함
    has_error = await error_elements.count() > 0
    has_response = await chat_messages.count() > 0
    
    assert has_error or has_response, "긴 입력에 대한 적절한 처리가 되지 않음"
    
    print("✅ 에러 처리 테스트 통과")

def main():
    """메인 함수"""
    print("🚀 Streamlit UI 자동 테스트 시작")
    print("=" * 50)
    
    try:
        # 비동기 테스트 실행
        asyncio.run(run_ui_tests())
        print("=" * 50)
        print("🎉 모든 테스트가 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ 테스트 실패: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)