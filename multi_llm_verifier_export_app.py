import os
import re
import ast
import sys
import subprocess
import tempfile
import uuid
import io
import zipfile
import json
import textwrap
from datetime import datetime
from typing import TypedDict, List, Literal, Dict, Any
import streamlit as st
import pandas as pd
import requests
import litellm
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# ReportLab PDF 라이브러리
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ==========================================
# 0. API 키 안전 조회 함수 (Streamlit Secrets / Session 격리 지원)
# ==========================================
def get_api_key(key_name: str) -> str:
    """
    보안 계층 구조:
    1. 사용자가 현재 브라우저 세션(사이드바)에 직접 입력한 키 (개별 격리, 세션 종료 시 소멸)
    2. Streamlit Cloud Secrets 대시보드 또는 로컬 .streamlit/secrets.toml
    3. 시스템 환경 변수 (os.environ)
    """
    # 1. 세션 스테이트 (사용자 입력)
    user_input = st.session_state.get(f"input_{key_name}", "")
    if user_input and str(user_input).strip():
        return str(user_input).strip()
    
    # 2. Streamlit Secrets (배포 서버 관리자 키)
    try:
        if key_name in st.secrets:
            return str(st.secrets[key_name]).strip()
    except Exception:
        pass

    # 3. 환경 변수
    env_val = os.environ.get(key_name, "")
    if env_val and env_val.strip():
        return env_val.strip()

    return ""

# ==========================================
# 1. 상태 정의 (AgentState)
# ==========================================
class AgentState(TypedDict):
    user_query: str
    raw_draft: str
    clean_content: str
    is_python_code: bool
    claude_feedback: str
    gpt_feedback: str
    status: str
    iteration: int
    max_iterations: int
    logs: List[str]
    cost_ledger: List[Dict[str, Any]]

# ==========================================
# 2. 모델 단가 및 비용 계산
# ==========================================
MODEL_PRICING_PER_1M = {
    "gemini/gemini-3.1-pro-preview": {"input": 1.25, "output": 5.00, "cache_discount": 0.75},
    "gemini/gemini-2.5-pro": {"input": 1.25, "output": 5.00, "cache_discount": 0.75},
    "gemini/gemini-2.5-flash": {"input": 0.15, "output": 0.60, "cache_discount": 0.75},
    "gemini/gemini-1.5-pro": {"input": 1.25, "output": 5.00, "cache_discount": 0.75},
    "gemini/gemini-1.5-flash": {"input": 0.075, "output": 0.30, "cache_discount": 0.75},
    "anthropic/claude-3-5-haiku-latest": {"input": 0.80, "output": 4.00, "cache_discount": 0.90},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cache_discount": 0.50},
}

def call_gemini_with_fallback(messages: list, temperature: float, api_key: str):
    """Google Gemini 최신 모델(3.1-pro-preview) 호출 및 모델 버전별 자동 폴백 지원"""
    candidate_models = [
        "gemini/gemini-3.1-pro-preview",
        "gemini/gemini-2.5-flash",
        "gemini/gemini-1.5-pro",
        "gemini/gemini-1.5-flash",
    ]
    last_err = None
    for m in candidate_models:
        try:
            resp = litellm.completion(
                model=m,
                messages=messages,
                temperature=temperature,
                api_key=api_key
            )
            return resp, m
        except litellm.NotFoundError as e:
            last_err = e
            continue
        except Exception as e:
            err_str = str(e).lower()
            if "not_found" in err_str or "no longer available" in err_str or "404" in err_str:
                last_err = e
                continue
            raise e
    if last_err:
        raise last_err


def extract_cost_entry(response: Any, step_name: str, model_name: str) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

    cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    if prompt_details and getattr(prompt_details, "cached_tokens", 0):
        cached_tokens += getattr(prompt_details, "cached_tokens", 0)

    pricing = MODEL_PRICING_PER_1M.get(model_name, {"input": 0.5, "output": 1.5, "cache_discount": 0.5})
    baseline_usd = (prompt_tokens / 1_000_000) * pricing["input"] + (completion_tokens / 1_000_000) * pricing["output"]
    saved_usd = (cached_tokens / 1_000_000) * pricing["input"] * pricing["cache_discount"]

    try:
        actual_usd = litellm.completion_cost(completion_response=response)
        if actual_usd == 0.0 and baseline_usd > 0:
            actual_usd = max(0.0, baseline_usd - saved_usd)
    except Exception:
        actual_usd = max(0.0, baseline_usd - saved_usd)

    is_gemini_free = st.session_state.get("gemini_free_tier", True)
    if "gemini" in model_name and is_gemini_free:
        saved_usd = baseline_usd
        actual_usd = 0.0

    exchange_rate = st.session_state.get("exchange_rate", 1380.0)
    
    return {
        "단계": step_name,
        "모델": model_name.split("/")[-1],
        "입력 토큰": prompt_tokens,
        "캐시 히트 토큰": cached_tokens,
        "출력 토큰": completion_tokens,
        "실제 비용(원)": round(actual_usd * exchange_rate, 3),
        "절약 금액(원)": round(saved_usd * exchange_rate, 3),
        "actual_krw_raw": actual_usd * exchange_rate,
        "saved_krw_raw": saved_usd * exchange_rate
    }

# ==========================================
# 3. PDF 생성 및 파일 내보내기 함수
# ==========================================
def register_korean_font() -> str:
    """Windows 및 주요 OS의 기본 한글 폰트를 자동 등록하여 글자 깨짐 방지"""
    candidates = [
        ("MalgunGothic", "C:/Windows/Fonts/malgun.ttf"),
        ("Gulim", "C:/Windows/Fonts/gulim.ttc"),
        ("Batang", "C:/Windows/Fonts/batang.ttc"),
        ("NanumGothic", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        ("AppleGothic", "/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
    ]
    for name, path in candidates:
        if os.path.exists(path):
            try:
                if name not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont(name, path))
                return name
            except Exception:
                continue
    return "Helvetica"

def build_export_pdf(state_vals: Dict[str, Any], ts_str: str) -> bytes:
    """최종 산출물을 손쉽게 열어볼 수 있도록 깔끔한 PDF 문서로 빌드"""
    font_name = register_korean_font()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    title_style = ParagraphStyle(
        "PDFTitle",
        fontName=font_name,
        fontSize=18,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "PDFSubTitle",
        fontName=font_name,
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=14
    )
    section_style = ParagraphStyle(
        "PDFSection",
        fontName=font_name,
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=12,
        spaceAfter=6
    )
    table_header_style = ParagraphStyle(
        "PDFTableHeader",
        fontName=font_name,
        fontSize=9,
        leading=12,
        textColor=colors.white
    )
    table_cell_style = ParagraphStyle(
        "PDFTableCell",
        fontName=font_name,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E293B")
    )
    code_block_style = ParagraphStyle(
        "PDFCodeBlock",
        fontName=font_name,
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#0F172A"),
        backColor=colors.HexColor("#F8FAFC"),
        borderColor=colors.HexColor("#CBD5E1"),
        borderWidth=1,
        borderPadding=8,
        spaceAfter=10
    )

    story = []

    # 1. 헤더 영역
    story.append(Paragraph("멀티 AI 모델 연합 교차 검증 최종 리포트", title_style))
    story.append(Paragraph(
        f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 엔진: Gemini 2.5 Pro + Claude 3.5 Haiku + ChatGPT (gpt-4o-mini)",
        subtitle_style
    ))

    # 2. 메타 요약 테이블
    user_query = state_vals.get("user_query", "미지정")
    status = state_vals.get("status", "PASS")
    iteration = state_vals.get("iteration", 0)
    ledger = state_vals.get("cost_ledger", [])
    total_cost = sum(item.get("actual_krw_raw", 0.0) for item in ledger)
    total_saved = sum(item.get("saved_krw_raw", 0.0) for item in ledger)

    summary_data = [
        [Paragraph("<b>항목</b>", table_header_style), Paragraph("<b>상세 내역</b>", table_header_style)],
        [Paragraph("사용자 원본 질의", table_cell_style), Paragraph(user_query, table_cell_style)],
        [Paragraph("최종 판정 상태", table_cell_style), Paragraph(f"<b>{status}</b>", table_cell_style)],
        [Paragraph("총 검증 루프 수", table_cell_style), Paragraph(f"{iteration} 회", table_cell_style)],
        [Paragraph("소모 비용 (원화)", table_cell_style), Paragraph(f"{total_cost:.2f} 원 (-{total_saved:.2f}원 절약)", table_cell_style)],
    ]
    summary_table = Table(summary_data, colWidths=[110, 412])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # 3. 최종 검증 완료 산출물
    content_type_str = "파이썬 코드" if state_vals.get("is_python_code") else "리포트 본문"
    story.append(Paragraph(f"최종 검증 완료 산출물 ({content_type_str})", section_style))
    raw_content = state_vals.get("clean_content", "").strip() or "출력 내용 없음"
    wrapped_lines = []
    for line in raw_content.split("\n"):
        if len(line) > 90:
            wrapped_lines.extend(textwrap.wrap(line, width=90, break_long_words=True))
        else:
            wrapped_lines.append(line)
    wrapped_code = "\n".join(wrapped_lines)
    story.append(Preformatted(wrapped_code, code_block_style))

    # 4. 단계별 감사 로그 요약
    logs = state_vals.get("logs", [])
    if logs:
        story.append(Spacer(1, 6))
        story.append(Paragraph("단계별 교차 검증 및 감사 로그", section_style))
        log_text = "\n".join(f"- {log}" for log in logs)
        story.append(Preformatted(log_text, code_block_style))

    doc.build(story)
    return buf.getvalue()

def build_export_zip(state_vals: Dict[str, Any], ts_str: str) -> bytes:
    buf = io.BytesIO()
    ext = "py" if state_vals.get("is_python_code") else "md"
    main_filename = f"verified_output_{ts_str}.{ext}"
    pdf_filename = f"verified_output_{ts_str}.pdf"
    
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        try:
            pdf_bytes = build_export_pdf(state_vals, ts_str)
            zf.writestr(pdf_filename, pdf_bytes)
        except Exception:
            pass

        zf.writestr(main_filename, state_vals.get("clean_content", ""))
        
        audit_lines = [
            f"# 🛡️ 멀티 LLM 교차 검증 감사 리포트 ({ts_str})",
            f"- **사용자 원본 질의**: {state_vals.get('user_query', '')}",
            f"- **최종 판정 상태**: {state_vals.get('status', '')}",
            f"- **총 외부 검증 루프**: {state_vals.get('iteration', 0)}회",
            "",
            "## 📋 단계별 실행 로그"
        ]
        for log in state_vals.get("logs", []):
            audit_lines.append(f"- {log}")
            
        audit_lines.extend([
            "",
            "## 🟣 마지막 Claude 1차 검증 피드백 (JSON)",
            "```json",
            str(state_vals.get("claude_feedback", "{}")),
            "```",
            "",
            "## 🟢 마지막 ChatGPT 최종 판정 피드백 (JSON)",
            "```json",
            str(state_vals.get("gpt_feedback", "{}")),
            "```"
        ])
        
        zf.writestr(f"verification_audit_report_{ts_str}.md", "\n".join(audit_lines))
        
        ledger = state_vals.get("cost_ledger", [])
        if ledger:
            df = pd.DataFrame(ledger).drop(columns=["actual_krw_raw", "saved_krw_raw"], errors="ignore")
            zf.writestr(f"cost_receipt_{ts_str}.csv", df.to_csv(index=False).encode("utf-8-sig"))
            
        zf.writestr(f"state_snapshot_{ts_str}.json", json.dumps(state_vals, ensure_ascii=False, indent=2, default=str))
        
    return buf.getvalue()

def upload_to_github_gist(filename: str, content: str, audit_summary: str, github_token: str, is_public: bool = False) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    payload = {
        "description": "Multi-LLM Verified Output (Gemini -> Claude -> ChatGPT)",
        "public": is_public,
        "files": {
            filename: {"content": content},
            "verification_summary.md": {"content": audit_summary}
        }
    }
    resp = requests.post("https://api.github.com/gists", headers=headers, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()

# ==========================================
# 4. LangGraph 5개 노드 구현 (정확한 아키텍처 매핑)
# ==========================================

# [Node 1: Gemini 초안 생성 (Heavy Lifter)]
def gemini_draft_node(state: AgentState) -> dict:
    step_label = "Node 1: Gemini 1차 초안 생성"
    prompt = state["user_query"]

    gemini_key = get_api_key("GEMINI_API_KEY")
    if not gemini_key:
        mock_code = (
            "import requests\n\n"
            "def fetch_user_data(user_id: int) -> dict:\n"
            "    url = f\"https://api.example.com/users/{user_id}\"\n"
            "    response = requests.get(url)  # 타임아웃 및 에러 검증 부재\n"
            "    return response.json()"
        )
        return {
            "raw_draft": f"```python\n{mock_code}\n```",
            "cost_ledger": state.get("cost_ledger", []) + [{
                "단계": step_label, "모델": "gemini-3.1-pro(Demo)",
                "입력 토큰": 150, "캐시 히트 토큰": 0, "출력 토큰": 350,
                "실제 비용(원)": 0.0, "절약 금액(원)": 3.3,
                "actual_krw_raw": 0.0, "saved_krw_raw": 3.3
            }],
            "logs": state.get("logs", []) + [f"🔷 [{step_label}] 완료 (초안 생성 완료)"]
        }

    response, used_model = call_gemini_with_fallback(
        messages=[
            {"role": "system", "content": "인사말이나 부연 설명 없이 요청한 코드 또는 본문만 즉시 작성하십시오. 코드인 경우 반드시 마크다운 코드 블록(```python ... ```)으로 감싸십시오."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        api_key=gemini_key
    )
    cost_entry = extract_cost_entry(response, step_label, used_model)
    content = response.choices[0].message.content or ""
    
    return {
        "raw_draft": content,
        "cost_ledger": state.get("cost_ledger", []) + [cost_entry],
        "logs": state.get("logs", []) + [f"🔷 [{step_label} ({used_model.split('/')[-1]})] 완료 (비용: {cost_entry['실제 비용(원)']:.2f}원)"]
    }

# [Node 2: Regex 전처리 필터] (인사말 제거, 핵심 코드/본문만 추출)
def regex_preprocess_node(state: AgentState) -> dict:
    text = state.get("raw_draft", "")
    py_blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    any_blocks = re.findall(r"```(?:\w+)?\s*\n(.*?)```", text, re.DOTALL)

    if py_blocks:
        clean_text, is_py = "\n\n".join(b.strip() for b in py_blocks), True
    elif any_blocks:
        clean_text, is_py = "\n\n".join(b.strip() for b in any_blocks), False
    else:
        clean_text = re.sub(r"^(네,|안녕하세요|요청하신|반갑습니다).*?\n+", "", text.strip(), flags=re.IGNORECASE)
        is_py = False

    return {
        "clean_content": clean_text,
        "is_python_code": is_py,
        "logs": state.get("logs", []) + [f"✂️ [Node 2: Regex 전처리 필터] 본문 정제 완료 ({len(clean_text):,}자 추출)"]
    }

# [Node 3: Claude 1차 검증] (Target/Replacement JSON 패치 생성)
def claude_validator_node(state: AgentState) -> dict:
    model_name = "anthropic/claude-3-5-haiku-latest"
    loop_num = state["iteration"] + 1
    step_label = f"Node 3: Claude 1차 검증 (L{loop_num})"

    claude_key = get_api_key("ANTHROPIC_API_KEY")
    if not claude_key:
        mock_feedback = json.dumps({
            "status": "MODIFY" if loop_num == 1 else "PASS",
            "modifications": [{
                "exact_target": "response = requests.get(url)",
                "replacement": "response = requests.get(url, timeout=10)\n    response.raise_for_status()",
                "reason": "타임아웃 설정 및 HTTP 상태코드 검증 추가"
            }] if loop_num == 1 else []
        }, ensure_ascii=False, indent=2)
        return {
            "claude_feedback": mock_feedback,
            "cost_ledger": state.get("cost_ledger", []) + [{
                "단계": step_label, "모델": "claude-3-5-haiku(Demo)",
                "입력 토큰": 520, "캐시 히트 토큰": 400, "출력 토큰": 95,
                "실제 비용(원)": 0.62, "절약 금액(원)": 0.40,
                "actual_krw_raw": 0.62, "saved_krw_raw": 0.40
            }],
            "logs": state.get("logs", []) + [f"🟣 [{step_label}] JSON 패치 생성 완료"]
        }

    sys_prompt = (
        '당신은 1차 코드/문서 검증기입니다. 인사말이나 설명 없이 오직 JSON 규격으로만 응답하십시오.\n'
        '응답 형식:\n'
        '{"status": "PASS" 또는 "MODIFY", "modifications": [{"exact_target": "원문 일부", "replacement": "수정문", "reason": "이유"}]}'
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt, "cache_control": {"type": "ephemeral"}}]},
        {"role": "user", "content": [{"type": "text", "text": f"[검증 대상 본문]\n{state['clean_content']}", "cache_control": {"type": "ephemeral"}}]}
    ]
    response = litellm.completion(
        model=model_name,
        messages=messages,
        max_tokens=600,
        temperature=0.0,
        api_key=claude_key
    )
    feedback = response.choices[0].message.content or "{}"
    cost_entry = extract_cost_entry(response, step_label, model_name)

    return {
        "claude_feedback": feedback,
        "cost_ledger": state.get("cost_ledger", []) + [cost_entry],
        "logs": state.get("logs", []) + [f"🟣 [{step_label}] 완료 (비용: {cost_entry['실제 비용(원)']:.2f}원)"]
    }

# [Node 4: ChatGPT 최종 검증 & 충돌 해결] (치명적 결함 위주 필터링 및 최종 JSON 확정)
def gpt_validator_node(state: AgentState) -> dict:
    model_name = "gpt-4o-mini"
    loop_num = state["iteration"] + 1
    step_label = f"Node 4: ChatGPT 최종 검증 (L{loop_num})"

    openai_key = get_api_key("OPENAI_API_KEY")
    if not openai_key:
        mock_feedback = state.get("claude_feedback", "{}")
        try:
            parsed = json.loads(mock_feedback)
            status = parsed.get("status", "PASS") if len(parsed.get("modifications", [])) > 0 else "PASS"
        except Exception:
            status = "PASS"
        return {
            "gpt_feedback": mock_feedback,
            "status": status,
            "iteration": loop_num,
            "cost_ledger": state.get("cost_ledger", []) + [{
                "단계": step_label, "모델": "gpt-4o-mini(Demo)",
                "입력 토큰": 615, "캐시 히트 토큰": 512, "출력 토큰": 88,
                "실제 비용(원)": 0.14, "절약 금액(원)": 0.06,
                "actual_krw_raw": 0.14, "saved_krw_raw": 0.06
            }],
            "logs": state.get("logs", []) + [f"🟢 [{step_label}] 최종 판정 확정: {status}"]
        }

    sys_prompt = (
        '당신은 최종 심판관 겸 검증기입니다. [검증 대상 본문]과 [Claude 1차 검증 내역]을 대조하여 '
        '치명적 결함 위주로 필터링하고 최종 JSON만 출력하십시오.\n'
        '응답 형식:\n'
        '{"status": "PASS" 또는 "MODIFY", "modifications": [{"exact_target": "원문 일부", "replacement": "수정문", "reason": "이유"}]}'
    )
    user_content = (
        f"[검증 대상 본문]\n{state['clean_content']}\n\n"
        f"[Claude 1차 검증 내역]\n{state['claude_feedback']}"
    )
    
    response = litellm.completion(
        model=model_name,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content}
        ],
        response_format={"type": "json_object"},
        max_tokens=600,
        temperature=0.0,
        api_key=openai_key
    )
    gpt_json_str = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(gpt_json_str)
        mods = parsed.get("modifications", [])
        status = parsed.get("status", "PASS") if len(mods) > 0 else "PASS"
    except Exception:
        status = "PASS"

    cost_entry = extract_cost_entry(response, step_label, model_name)

    return {
        "gpt_feedback": gpt_json_str,
        "status": status,
        "iteration": loop_num,
        "cost_ledger": state.get("cost_ledger", []) + [cost_entry],
        "logs": state.get("logs", []) + [f"🟢 [{step_label}] 최종 판정: {status} (비용: {cost_entry['실제 비용(원)']:.2f}원)"]
    }

# [Node 5: Gemini 패치 병합]
def gemini_merge_node(state: AgentState) -> dict:
    step_label = f"Node 5: Gemini 패치 병합 (L{state['iteration']})"

    gemini_key = get_api_key("GEMINI_API_KEY")
    if not gemini_key:
        clean_text = state.get("clean_content", "")
        try:
            parsed = json.loads(state.get("gpt_feedback", "{}"))
            for mod in parsed.get("modifications", []):
                t = mod.get("exact_target", "")
                r = mod.get("replacement", "")
                if t and t in clean_text:
                    clean_text = clean_text.replace(t, r)
        except Exception:
            pass

        return {
            "clean_content": clean_text,
            "cost_ledger": state.get("cost_ledger", []) + [{
                "단계": step_label, "모델": "gemini-3.1-pro(Demo)",
                "입력 토큰": 680, "캐시 히트 토큰": 500, "출력 토큰": 480,
                "실제 비용(원)": 0.0, "절약 금액(원)": 4.28,
                "actual_krw_raw": 0.0, "saved_krw_raw": 4.28
            }],
            "logs": state.get("logs", []) + [f"🔷 [{step_label}] 패치 병합 완료 -> Node 3(Claude) 재검증 순환"]
        }

    prompt = (
        "아래 [원본 산출물]에서 [최종 승인된 수정 지시서(JSON)]의 exact_target을 찾아 replacement로 정확히 교체하여 "
        "완성된 전체 산출물만 출력하십시오. 코드일 경우 ```python 블록으로 감싸십시오.\n\n"
        f"[원본 산출물]\n{state['clean_content']}\n\n"
        f"[수정 지시서]\n{state['gpt_feedback']}"
    )
    response, used_model = call_gemini_with_fallback(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        api_key=gemini_key
    )
    merged_raw = response.choices[0].message.content or ""
    cost_entry = extract_cost_entry(response, step_label, used_model)

    py_blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", merged_raw, re.DOTALL | re.IGNORECASE)
    any_blocks = re.findall(r"```(?:\w+)?\s*\n(.*?)```", merged_raw, re.DOTALL)
    if py_blocks:
        clean_text, is_py = "\n\n".join(b.strip() for b in py_blocks), True
    elif any_blocks:
        clean_text, is_py = "\n\n".join(b.strip() for b in any_blocks), False
    else:
        clean_text = re.sub(r"^(네,|안녕하세요|요청하신|반갑습니다).*?\n+", "", merged_raw.strip(), flags=re.IGNORECASE)
        is_py = state.get("is_python_code", False)

    return {
        "raw_draft": merged_raw,
        "clean_content": clean_text,
        "is_python_code": is_py,
        "cost_ledger": state.get("cost_ledger", []) + [cost_entry],
        "logs": state.get("logs", []) + [f"🔷 [{step_label}] 패치 병합 완료 -> Node 3(Claude) 재검증 루프 순환 (비용: {cost_entry['실제 비용(원)']:.2f}원)"]
    }

# ==========================================
# 5. 조건부 라우터 및 그래프 컴파일 (워크플로우 매핑)
# ==========================================
def route_after_gpt(state: AgentState) -> Literal["gemini_merge", "__end__"]:
    if state["status"] == "PASS" or state["iteration"] >= state["max_iterations"]:
        return END
    return "gemini_merge"

def get_compiled_graph(enable_hitl: bool = False):
    builder = StateGraph(AgentState)
    builder.add_node("gemini_draft", gemini_draft_node)
    builder.add_node("regex_preprocess", regex_preprocess_node)
    builder.add_node("claude_validate", claude_validator_node)
    builder.add_node("gpt_validate", gpt_validator_node)
    builder.add_node("gemini_merge", gemini_merge_node)
    
    # 엣지 연결
    builder.add_edge(START, "gemini_draft")
    builder.add_edge("gemini_draft", "regex_preprocess")
    builder.add_edge("regex_preprocess", "claude_validate")
    builder.add_edge("claude_validate", "gpt_validate")
    builder.add_conditional_edges(
        "gpt_validate",
        route_after_gpt,
        {"gemini_merge": "gemini_merge", END: END}
    )
    builder.add_edge("gemini_merge", "claude_validate")
    
    memory = MemorySaver()
    interrupts = ["gemini_merge"] if enable_hitl else []
    return builder.compile(checkpointer=memory, interrupt_before=interrupts)

# ==========================================
# 6. UI 대시보드 렌더링 함수
# ==========================================
def render_cost_dashboard(ledger: List[Dict[str, Any]]):
    if not ledger:
        return
    
    total_actual = sum(item["actual_krw_raw"] for item in ledger)
    total_saved = sum(item["saved_krw_raw"] for item in ledger)
    total_in_tokens = sum(item["입력 토큰"] for item in ledger)
    total_cached_tokens = sum(item["캐시 히트 토큰"] for item in ledger)
    total_out_tokens = sum(item["출력 토큰"] for item in ledger)
    cache_ratio = (total_cached_tokens / total_in_tokens * 100) if total_in_tokens > 0 else 0.0

    st.markdown("### 🧾 실시간 API 토큰 & 원화(KRW) 비용 영수증")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("💰 이번 질의 소모 비용", f"{total_actual:.2f} 원", f"-{total_saved:.2f} 원 절약")
    m2.metric("⚡ 캐시 및 무료티어 절약액", f"{total_saved:.2f} 원")
    m3.metric("📥 총 입력 토큰 (캐시율)", f"{total_in_tokens:,} 개", f"캐시: {total_cached_tokens:,}개 ({cache_ratio:.1f}%)")
    m4.metric("📤 총 출력 토큰 (JSON 압축)", f"{total_out_tokens:,} 개")

    with st.expander("📊 단계별 상세 과금 내역서 (클릭하여 열기)", expanded=False):
        df = pd.DataFrame(ledger).drop(columns=["actual_krw_raw", "saved_krw_raw"], errors="ignore")
        st.dataframe(df, use_container_width=True, hide_index=True)

def render_export_center(state_vals: Dict[str, Any]):
    st.markdown("### 💾 산출물 원클릭 다운로드 & 클라우드 백업 센터")
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    is_py = state_vals.get("is_python_code", False)
    raw_ext = "py" if is_py else "md"
    raw_mime = "text/x-python" if is_py else "text/markdown"
    
    default_pdf_fname = f"verified_result_{ts_str}.pdf"
    default_raw_fname = f"verified_result_{ts_str}.{raw_ext}"
    content_str = state_vals.get("clean_content", "")
    
    with st.spinner("📄 PDF 리포트 빌드 중..."):
        try:
            pdf_bytes = build_export_pdf(state_vals, ts_str)
        except Exception as e:
            pdf_bytes = b""
            st.warning(f"PDF 생성 중 예외 발생: {e}")
            
    zip_bytes = build_export_zip(state_vals, ts_str)

    ledger = state_vals.get("cost_ledger", [])
    csv_bytes = b""
    if ledger:
        df_cost = pd.DataFrame(ledger).drop(columns=["actual_krw_raw", "saved_krw_raw"], errors="ignore")
        csv_bytes = df_cost.to_csv(index=False).encode("utf-8-sig")

    st.success("📕 **PDF 문서 준비 완료!** 아래 버튼을 누르면 PC에서 프로그램 설치 없이 브라우저나 기본 뷰어로 즉시 열람하실 수 있습니다.")

    c_pdf, c_raw, c_zip, c_csv = st.columns(4)
    with c_pdf:
        st.download_button(
            label="📕 최종 산출물 다운로드 (.pdf)",
            data=pdf_bytes,
            file_name=default_pdf_fname,
            mime="application/pdf",
            type="primary",
            use_container_width=True,
            help="한글 폰트, 코드 블록, 검증 통계가 깔끔하게 정돈되어 쉽게 열리는 PDF 파일입니다."
        )
    with c_raw:
        st.download_button(
            label=f"📄 원본 파일 다운로드 (.{raw_ext})",
            data=content_str.encode("utf-8"),
            file_name=default_raw_fname,
            mime=raw_mime,
            use_container_width=True,
            help="개발자용 순수 파이썬/마크다운 소스 코드 파일입니다."
        )
    with c_zip:
        st.download_button(
            label="📦 전체 패키지 다운로드 (.zip)",
            data=zip_bytes,
            file_name=f"verification_bundle_{ts_str}.zip",
            mime="application/zip",
            help="PDF + 원본 파일 + 교차검증 감사 리포트(.md) + 비용 영수증(.csv) + 상태 스냅샷(.json) 일괄 압축",
            use_container_width=True
        )
    with c_csv:
        st.download_button(
            label="🧾 비용 영수증 다운로드 (.csv)",
            data=csv_bytes,
            file_name=f"cost_receipt_{ts_str}.csv",
            mime="text/csv",
            disabled=(len(ledger) == 0),
            use_container_width=True
        )

    github_token = get_api_key("GITHUB_GIST_TOKEN")
    if github_token:
        st.divider()
        if st.button("🐙 GitHub Gist 자동 백업", use_container_width=True):
            try:
                audit_summary = "\n".join(state_vals.get("logs", []))
                res_json = upload_to_github_gist(
                    filename=default_raw_fname,
                    content=content_str,
                    audit_summary=audit_summary,
                    github_token=github_token,
                    is_public=st.session_state.get("gist_public", False)
                )
                st.success(f"Gist 백업 완료! [열기]({res_json.get('html_url')})")
            except Exception as e:
                st.error(f"Gist 업로드 실패: {e}")

# ==========================================
# 7. Streamlit 메인 화면 구성
# ==========================================
st.set_page_config(page_title="Multi-LLM Cross-Verification Workstation", layout="wide", page_icon="🛡️")
st.title("🛡️ 멀티 AI 모델 연합 교차 검증 워크스테이션")

with st.sidebar:
    st.header("🔑 LLM API 키 설정")
    
    # Secrets 등록 여부 확인
    secrets_gemini = False
    try:
        secrets_gemini = "GEMINI_API_KEY" in st.secrets or bool(os.environ.get("GEMINI_API_KEY"))
    except Exception:
        pass

    secrets_claude = False
    try:
        secrets_claude = "ANTHROPIC_API_KEY" in st.secrets or bool(os.environ.get("ANTHROPIC_API_KEY"))
    except Exception:
        pass

    secrets_openai = False
    try:
        secrets_openai = "OPENAI_API_KEY" in st.secrets or bool(os.environ.get("OPENAI_API_KEY"))
    except Exception:
        pass

    gemini_key_input = st.text_input(
        "Gemini API Key (Google AI Studio)" + (" (서버 등록됨)" if secrets_gemini else ""),
        type="password",
        key="input_GEMINI_API_KEY",
        placeholder="등록된 서버 키 사용 중" if secrets_gemini else "AIzaSy..."
    )
    claude_key_input = st.text_input(
        "Anthropic API Key" + (" (서버 등록됨)" if secrets_claude else ""),
        type="password",
        key="input_ANTHROPIC_API_KEY",
        placeholder="등록된 서버 키 사용 중" if secrets_claude else "sk-ant-..."
    )
    openai_key_input = st.text_input(
        "OpenAI API Key" + (" (서버 등록됨)" if secrets_openai else ""),
        type="password",
        key="input_OPENAI_API_KEY",
        placeholder="등록된 서버 키 사용 중" if secrets_openai else "sk-..."
    )

    has_all_keys = bool(get_api_key("GEMINI_API_KEY") and get_api_key("ANTHROPIC_API_KEY") and get_api_key("OPENAI_API_KEY"))
    if has_all_keys:
        st.success("🟢 모든 API 키가 준비되었습니다. (실시간 연합 교차 검증)")
    else:
        st.info("💡 API 키가 없는 경우에도 '데모 시뮬레이션 모드'로 파이프라인 전체 순환 과정을 직접 테스트할 수 있습니다.")

    st.divider()
    st.header("⚙️ 파이프라인 실행 모드")
    enable_hitl = st.checkbox("⏸️ 단계별 수동 승인 모드 (Human-in-the-Loop)", value=False,
                              help="체크 시 ChatGPT 검증 후 패치를 사용자가 직접 수정/승인한 뒤 Gemini 병합을 진행합니다. 체크 해제 시 다이어그램에 따른 '완전 자동 연속 루프'로 작동합니다.")
    max_loops = st.slider("최대 외부 검증 루프 횟수 (max_loops)", 1, 5, 2)

    st.divider()
    st.header("💱 환율 및 비용 설정")
    st.session_state["exchange_rate"] = st.number_input("적용 환율 (1 USD 기준 원화)", value=1380.0, step=10.0)
    st.session_state["gemini_free_tier"] = st.checkbox("Gemini 무료 티어 적용 (비용 0원 처리)", value=True)

    st.divider()
    st.header("🐙 GitHub Gist 백업 설정 (선택)")
    st.text_input(
        "GitHub Personal Access Token (gist 권한)", type="password",
        key="input_GITHUB_GIST_TOKEN",
        help="입력 시 최종 산출물을 내 GitHub Gist로 원클릭 전송할 수 있습니다."
    )
    st.session_state["gist_public"] = st.checkbox("공개(Public) Gist로 생성", value=False)

# 워크플로우 아키텍처 다이어그램 표시
with st.expander("🗺️ 연합 교차 검증 파이프라인 아키텍처 다이어그램", expanded=False):
    st.markdown("""
```mermaid
flowchart TD
    START([START]) --> N1["Node 1: Gemini 초안 생성 (Heavy Lifter)"]
    N1 --> N2["Node 2: Regex 전처리 필터<br/>(인사말 제거, 핵심 코드/본문만 추출)"]
    N2 --> N3["Node 3: Claude 1차 검증<br/>(Target/Replacement JSON 패치 생성)"]
    N3 --> N4["Node 4: ChatGPT 최종 검증 & 충돌 해결<br/>(치명적 결함 위주 필터링 및 최종 JSON 확정)"]
    N4 --> CR{"Conditional Router<br/>조건부 분기"}
    CR -- "수정 사항 존재 시<br/>(status == 'MODIFY' & loop < max_loops)" --> N5["Node 5: Gemini 패치 병합"]
    N5 --> N3
    CR -- "status == 'PASS' 또는<br/>최대 루프 도달 시" --> FINISH([END<br/>최종 산출물 반환])
```
""")

graph = get_compiled_graph(enable_hitl=enable_hitl)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": st.session_state.thread_id}}

user_query = st.chat_input("작성할 파이썬 코드나 리포트 주제를 입력하세요 (예: 비동기 웹 크롤러 파이썬 코드 작성해줘)...")
if user_query:
    st.session_state.thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    initial_state: AgentState = {
        "user_query": user_query,
        "raw_draft": "",
        "clean_content": "",
        "is_python_code": False,
        "claude_feedback": "",
        "gpt_feedback": "",
        "status": "MODIFY",
        "iteration": 0,
        "max_iterations": max_loops,
        "logs": [],
        "cost_ledger": []
    }
    with st.spinner("🚀 멀티 AI 모델 연합 교차 검증 파이프라인 가동 중..."):
        try:
            for _ in graph.stream(initial_state, config, stream_mode="values"):
                pass
        except Exception as e:
            st.error(f"파이프라인 실행 중 오류가 발생했습니다: {e}")

current_snapshot = graph.get_state(config)

if current_snapshot.values:
    state_vals = current_snapshot.values
    
    render_cost_dashboard(state_vals.get("cost_ledger", []))
    st.divider()

    with st.expander("📋 파이프라인 진행 로그 보기", expanded=True):
        for log in state_vals.get("logs", []):
            st.write(log)

    # Human-in-the-Loop 대기 화면
    if "gemini_merge" in current_snapshot.next:
        st.warning("⏸️ **[Human-in-the-Loop 대기 중]** ChatGPT가 확정 패치를 제안했습니다. 승인 시 Node 5: Gemini 병합이 실행됩니다.")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("현재 초안 본문")
            st.code(state_vals.get("clean_content", ""), language="python" if state_vals.get("is_python_code") else "markdown")
        with col2:
            st.subheader("ChatGPT 확정 수정 지시서 (편집 가능)")
            edited_feedback = st.text_area("JSON 패치 검토 및 수정:", value=state_vals.get("gpt_feedback", "{}"), height=250)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ 승인 및 Node 5 병합 진행", type="primary", use_container_width=True):
                    graph.update_state(config, {"gpt_feedback": edited_feedback})
                    with st.spinner("제미나이가 패치를 병합하고 Claude 재검증을 수행 중입니다..."):
                        for _ in graph.stream(None, config, stream_mode="values"):
                            pass
                    st.rerun()
            with c2:
                if st.button("🛑 수정 거부 및 즉시 확정 (비용 방어)", key="reject_btn", use_container_width=True):
                    graph.update_state(config, {"status": "PASS"})
                    st.rerun()

    elif not current_snapshot.next or state_vals.get("status") == "PASS":
        st.success(f"🏁 **최종 교차 검증 완료!** (외부 순환 검증: {state_vals.get('iteration', 0)}회 완료, 최종 판정: {state_vals.get('status', 'PASS')})")
        
        render_export_center(state_vals)
        st.divider()
        
        st.subheader("📄 최종 검증 완료 산출물 미리보기")
        if state_vals.get("is_python_code"):
            st.code(state_vals.get("clean_content", ""), language="python")
        else:
            st.markdown(state_vals.get("clean_content", ""))
