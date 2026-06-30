import streamlit as st
import os
import glob
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# PyPDF2 대신 환경에 맞춘 최신 pypdf 사용
try:
    from pypdf import PdfReader
except ImportError:
    st.error("⚠️ pypdf 라이브러리가 설치되지 않았습니다. requirements.txt를 확인해주세요.")

# [로고 적용] 탭 아이콘 변경
st.set_page_config(page_title="AI 식품/축산물 규제 검토 시스템", page_icon="yonsei_logo.png", layout="wide")

# --- 💡 헤더 및 로고 배치 ---
title_col1, title_col2 = st.columns([0.7, 9.3])
with title_col1:
    try:
        st.image("yonsei_logo.png", width=70) 
    except:
        st.title("🏢") 

with title_col2:
    st.title("연세유업 규제 및 행정처분 AI 검색 시스템")

st.markdown("""
**식품·축산물 관련 법령 및 행정처분 기준을 신속하고 정확하게 조회하기 위한 사내 AI 법무 검토 솔루션입니다.** *(※ 식약처 고시, 법령 원문(PDF) 및 사내 처분기준표(Excel)를 교차 검증하여 환각(오류) 없는 신뢰도 높은 리포트를 도출합니다.)*
""")

try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("⚠️ 시스템 설정(Secrets)에 GOOGLE_API_KEY가 등록되지 않았습니다.")
    st.stop()

# --- 💡 세션 상태 초기화 ---
if 'phase' not in st.session_state:
    st.session_state.phase = 1
if 'keyword_options' not in st.session_state:
    st.session_state.keyword_options = []
if 'direction_options' not in st.session_state:
    st.session_state.direction_options = []
if 'case_options' not in st.session_state:
    st.session_state.case_options = []
if 'business_type' not in st.session_state:
    st.session_state.business_type = "🌐 공통 적용 (업종 무관)"

# --- 💡 업종별 동적 DB 로딩 (네거티브 필터링 방식) ---
@st.cache_data(show_spinner="사내 데이터베이스(관련 법령 및 행정처분 기준표)를 동기화 중입니다...")
def load_filtered_documents(biz_type):
    combined_text = "==== [연세유업 마스터 통합 데이터베이스 (엑셀+PDF)] ====\n\n"
    
    all_files = glob.glob("*.xlsx") + glob.glob("*.pdf")
    filtered_files = []
    
    for f in all_files:
        fname = f.lower()
        
        # [블랙리스트 방식 필터링]
        if "식품제조" in biz_type:
            if "축산물" not in fname:
                filtered_files.append(f)
        elif "축산물" in biz_type:
            if "식품위생법" not in fname:
                filtered_files.append(f)
        else:
            filtered_files.append(f)
            
    filtered_files = list(set(filtered_files))

    for file in [f for f in filtered_files if f.endswith('.xlsx')]:
        try:
            df = pd.read_excel(file)
            combined_text += f"\n--- [엑셀 문서: {file}] ---\n"
            combined_text += df.to_markdown(index=False) + "\n\n"
        except Exception as e:
            st.error(f"⚠️ {file} (엑셀) 읽기 실패: {e}")
            
    for file in [f for f in filtered_files if f.endswith('.pdf')]:
        try:
            with open(file, 'rb') as f:
                reader = PdfReader(f)
                combined_text += f"\n--- [PDF 문서: {file}] ---\n"
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        combined_text += text + "\n"
                combined_text += "\n\n"
        except Exception as e:
            st.error(f"⚠️ {file} (PDF) 읽기 실패: {e}")
            
    if not filtered_files:
        return "⚠️ 로딩된 데이터 파일이 없습니다. 시스템 폴더에 .xlsx 또는 .pdf 파일을 업로드해 주십시오."
        
    return combined_text

# --- 💡 프롬프트 정의 ---
KEYWORD_TEMPLATE = """
당신은 현장 용어를 법률 용어로 번역하는 AI입니다.
사용자의 질문에서 고유명사를 제거하고, 법전에서 검색될 만한 '3가지 다른 뉘앙스의 법률 키워드(행위 본질)' 옵션을 제안하십시오.
절대 부연 설명 없이, 숫자 1, 2, 3으로 시작하는 3줄의 텍스트만 출력하십시오.

사용자 질문: {question}
"""

DIRECTION_TEMPLATE = """
당신은 연세유업의 수석 법무 검토관이자 현장 식약처 감사(Audit) 전문가입니다.
사용자의 질문과 선택된 '법률 키워드'를 결합하여, 이를 처벌할 수 있는 '3가지 법률 적용 방향(관점)'을 제안하십시오.

🚨 [절대 준수 규칙] 🚨
1. [품목 기반 법령 자동 분류]: 우유/치즈는 「축산물 위생관리법」, 두유/과채는 「식품위생법」 관점 우선 제안.
2. [현장 감사 관점]: 표면적 위반 외에 파생 위반(성분배합비율, 서류 허위작성 등) 관점 반드시 포함.
3. 부연 설명 없이 숫자 1, 2, 3으로 시작하는 3줄의 텍스트만 출력하십시오.

사용자 질문: {question}
선택된 키워드: {selected_keyword}
"""

CASE_TEMPLATE = """
당신은 데이터베이스 원문 분석 및 요약 전담 AI입니다.
사용자의 질문, [업종], [위반 구역]을 바탕으로 제공된 [마스터 통합 데이터베이스]에서 관련 조항을 찾으십시오.

🚨 [가독성, 스포일러 방지 및 🔀 갈림길 질문 규칙] 🚨
1. 실무자가 한눈에 읽을 수 있도록 요약된 형태로 3~5개 제안하십시오.
2. ⚠️ [처분 수위 노출 금지]: 5단계에서는 "영업정지 1개월" 같은 처분 결과를 절대 텍스트에 노출하지 마십시오.
3. 🔀 [조건 분기 (갈림길 질문) 강제]: 사용자의 상황이 '내부 공정용'인지 '외부 판매용'인지 등에 따라 적용되는 법조항과 처분 수위가 극명하게 갈리는 경우, 임의로 하나만 선택하지 마십시오. 반드시 `[내부 사용 목적]`, `[외부 반출/판매 목적]`과 같이 상황을 분리하여 각각 별도의 보기로 제시하십시오. 실무자가 이를 보고 본인의 팩트에 맞는 것을 직접 선택(답변)하게 해야 합니다.
4. 🎯 [AI 추천 마크]: 가장 질문 정황에 부합하고 리스크가 큰 조항 하나에 `[⭐ AI 강력 추천]` 마크를 달아주십시오.
5. 부연 설명 없이 숫자 1, 2, 3으로 시작하십시오.

[마스터 통합 데이터베이스 (엑셀+PDF)]:
{db_data}

사용자 질문: {question}
선택된 법률 방향: {selected_direction}
강제 지정된 업종: {business_type}
강제 지정된 위반 구역: {selected_category}
"""

TEMPLATE = """
당신은 연세유업의 데이터베이스 통합 추출 전담 AI입니다.
실무자가 선택한 요약된 위반 상황('{selected_case}')을 바탕으로, [마스터 통합 데이터베이스]로 다시 돌아가 정확한 팩트를 찾아 리포트를 작성하십시오. (선택지 앞의 [⭐ AI 강력 추천] 텍스트나 [조건] 태그는 검색 시 무시하십시오.)

🚨 [최종 리포트 도출 3단계 검증 프로세스] 🚨

▶ **[Pass 1: DB 원문 역추적 및 팩트 추출]**
1. 선택된 요약본('{selected_case}')과 의미가 100% 일치하는 실제 엑셀/PDF 원문 데이터를 찾으십시오.
2. ⚠️ [법령 물리적 격리]: [{business_type}]이 '일반 식품'을 포함하면 「축산물 위생관리법」 관련 조항을 절대 출력하지 마십시오.

▶ **[Pass 1.5: 파생 위반 리스크 팩트체크]**
장부 조작 등 연쇄 적발될 수 있는 조항을 찾되, [법령 물리적 격리]를 준수하십시오.

---FINAL_REPORT---

▶ **[Pass 2: 최종 리포트 도출 (출력)]**
가독성을 위해 표를 사용하지 말고, 아래 제시된 강조형 리스트 포맷을 정확히 지켜서 출력하십시오. HTML 태그(br 등) 절대 사용 금지.

### 📊 위반 사항 및 행정처분 요약

* **1. 위반 의심 사항:** (질문 요약 및 위반 행위 팩트 기재)
* **2. 관련 법령 및 조항:** (DB 원문에 명시된 공식 조항 번호 기재)
* **3. 위반사항 (원문):** > (역추적하여 찾아낸 DB 내 실제 텍스트 원문 전체 복사. 요약 금지)
* **4. 행정처분 수위 및 정확한 출처:** [1차] 내용 / [2차] 내용 / [3차] 내용 (※ 처분 근거: OOO법 시행규칙 [별표 OO] 제O호 등 DB에서 찾은 가장 정확한 출처를 반드시 명시하여 실무자가 직접 법전을 찾아볼 수 있게 할 것)
* **5. 과태료/과징금:** (DB에서 확인된 액수 팩트만 기재, 없으면 '해당 없음')

---
### 🚨 [중요] 연쇄 파생 위반 리스크
(표면적 위반 외에 연쇄 적발될 수 있는 조항 및 처분 수위를 글머리기호(-)로 나열. 여기에도 괄호 열고 시행규칙 [별표 OO] 등 근거를 짧게 명시할 것)

---
### 💡 품질관리 및 대응 가이드
(글머리기호(1. 2. 3.)를 사용하여 구체적 대처 방안 나열)

[마스터 통합 데이터베이스]:
{db_data}

---
사용자 질문: {question}
강제 지정된 업종: {business_type}
선택된 위반 구역: {selected_category}
실무자가 선택한 요약 상황: {selected_case}
"""

# --- 💡 UI 구성 ---
user_question = st.text_area("🔍 검토가 필요한 위반 의심 사례, 표시사항 누락 등 구체적인 상황을 입력해 주십시오:", height=100)

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("▶ 1단계: 법률 키워드 추출", type="primary"):
        if user_question.strip():
            with st.spinner("AI가 상황을 분석하여 법률 키워드를 추출하고 있습니다..."):
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0.2)
                st.session_state.keyword_options = [opt.strip() for opt in (PromptTemplate.from_template(KEYWORD_TEMPLATE) | llm | StrOutputParser()).invoke({"question": user_question}).split('\n') if opt.strip() and opt[0].isdigit()]
                st.session_state.phase = 2

# 2단계
if st.session_state.phase >= 2 and st.session_state.keyword_options:
    st.markdown("### 🎯 2단계: 분석 대상 법률 키워드 선택")
    selected_kw = st.radio("적용할 핵심 키워드 지정:", st.session_state.keyword_options, key="kw_radio")
    
    if st.button("▶ 3단계: 법률 적용 관점 분석", type="secondary"):
        with st.spinner("해당 키워드를 바탕으로 법률 쟁점을 분석 중입니다..."):
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0.2)
            st.session_state.direction_options = [opt.strip() for opt in (PromptTemplate.from_template(DIRECTION_TEMPLATE) | llm | StrOutputParser()).invoke({"question": user_question, "selected_keyword": selected_kw}).split('\n') if opt.strip() and opt[0].isdigit()]
            st.session_state.phase = 3

# 3단계
if st.session_state.phase >= 3 and st.session_state.direction_options:
    st.markdown("### 🏛️ 3단계: 검토 대상 법률 확정")
    selected_dir = st.radio("검토를 진행할 법률 관점 지정:", st.session_state.direction_options, key="dir_radio")
    
    if st.button("▶ 4단계: 업종 및 위반 유형 지정", type="secondary"):
        st.session_state.phase = 4

# 4단계
if st.session_state.phase >= 4:
    st.markdown("---")
    st.markdown("### 🗂️ 4단계: 검색 대상 업종 및 위반 유형 고정")
    
    st.markdown("#### ① 대상 업종 선택")
    biz_choices = [
        "🏢 식품제조·가공업 (일반 식품, 두유, 과채음료 등)",
        "🥩 축산물가공업 / 식육포장처리업 (유가공품, 우유, 치즈 등)",
        "🏪 즉석판매제조·가공업 / 식품접객업",
        "🌐 공통 적용 (업종 무관)"
    ]
    st.session_state.business_type = st.radio("처분 대상 업종:", biz_choices, key="biz_radio")

    st.markdown("#### ② 위반 쟁점 카테고리 선택")
    category_choices = [
        "🛑 [위생/안전/이물] 금속, 벌레 혼입, 식중독균, 위생적 취급기준 위반",
        "📝 [서류/장부/보고] 품목제조보고 위반, 생산일지 및 원료수불부 허위작성",
        "🔬 [기준/규격/검사] 성분배합비율 위반, 자가품질검사 미실시",
        "📋 [기본 표시사항] 필수 정보 누락 및 완전 무표시",
        "⚠️ [안전/주의 표시] 알레르기, 당알코올 등 소비자 안전 주의문구 누락",
        "📢 [부당한 표시·광고] 오인·혼동, 허위/과대 광고",
        "🔍 기타 위반 (직접 입력)"
    ]
    selected_cat = st.radio("위반 카테고리:", category_choices, key="cat_radio")

    if selected_cat == "🔍 기타 위반 (직접 입력)":
        custom_cat = st.text_input("💡 엑셀에서 집중적으로 검색할 키워드를 직접 입력해 주십시오 (예: 건강진단, 지하수 등):")
        if custom_cat:
            selected_cat = custom_cat

    if st.button("▶ 5단계: 조항 및 처분기준 조회", type="secondary"):
        with st.spinner("선택된 업종에 맞는 사내 DB만을 필터링하여 검색 중입니다..."):
            current_db_data = load_filtered_documents(st.session_state.business_type)
            
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0)
            st.session_state.case_options = [opt.strip() for opt in (PromptTemplate.from_template(CASE_TEMPLATE) | llm | StrOutputParser()).invoke({
                "db_data": current_db_data,
                "question": user_question,
                "selected_direction": selected_dir,
                "business_type": st.session_state.business_type,
                "selected_category": selected_cat
            }).split('\n') if opt.strip() and opt[0].isdigit()]
            st.session_state.phase = 5

# 5단계
if st.session_state.phase == 5 and st.session_state.case_options:
    st.markdown("### 📋 5단계: 세부 위반 조항 최종 확인 (위반행위 요약)")
    st.info("💡 처분 수위가 가장 무겁고 질문 상황에 적합한 조항에 **[⭐ AI 강력 추천]** 마크가 표시되며, 상황 분기(예: 내부/외부용)가 필요한 경우 선택지가 나뉘어 출력됩니다.")
    selected_case = st.radio("해당 위반 조항:", st.session_state.case_options, key="case_radio")

    if st.button("📄 최종 법무 검토 및 행정처분 리포트 생성", type="primary"):
        with st.spinner("최종 리포트를 산출하고 있습니다..."):
            current_db_data = load_filtered_documents(st.session_state.business_type)
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0)
            rag_chain = PromptTemplate.from_template(TEMPLATE) | llm | StrOutputParser()
            
            full_response = rag_chain.invoke({
                "db_data": current_db_data,
                "question": user_question, 
                "business_type": st.session_state.business_type,
                "selected_category": selected_cat,
                "selected_case": selected_case
            })
            
        st.markdown("### 📊 최종 법무 검토 및 행정처분 리포트")
        if "---FINAL_REPORT---" in full_response:
            reasoning_part, report_part = full_response.split("---FINAL_REPORT---", 1)
            with st.expander("🔍 [참고] AI 교차 검색 로그 및 원문 팩트체크 내역"):
                st.markdown(reasoning_part.strip())
            st.markdown(report_part.strip())
        else:
            st.markdown(full_response)
