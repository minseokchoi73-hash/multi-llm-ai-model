@echo off
chcp 65001 > nul
title 멀티 AI 모델 연합 교차 검증 워크스테이션
echo ========================================================
echo [1/2] Python 환경 및 필수 패키지 점검 중...
echo ========================================================
python -c "import streamlit, langgraph, litellm" 2>nul
if %errorlevel% neq 0 (
    echo [안내] 필수 라이브러리 설치가 필요합니다. 설치를 진행합니다...
    if not exist venv (
        echo 가상환경(venv)을 생성합니다...
        python -m venv venv
    )
    call venv\Scripts\activate
    python -m pip install -r requirements.txt
) else (
    echo [성공] Streamlit, LangGraph, LiteLLM 등 모든 필수 패키지가 준비되어 있습니다.
)

echo ========================================================
echo [2/2] 멀티 LLM 교차 검증 웹앱을 실행합니다! (브라우저 자동 오픈)
echo ========================================================
python -m streamlit run multi_llm_verifier_export_app.py
pause
