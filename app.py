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
if 'db_data' not in st.session_state:
    st.session_state.db_data = ""

# --- 💡 엑셀(.xlsx) + PDF 통합 DB 로딩 ---
@st.cache_data(show_spinner="사내 데이터베이스(관련 법령 및 행정처분 기준표)를 동기화 중입니다...")
def load_all_documents():
    combined_text = "==== [연세유업 마스터 통합 데이터베이스 (엑셀+PDF)] ====\n\n"
    
    excel_files = glob.glob("*.xlsx")
    for file in excel_files:
        try:
            df = pd.read_excel(file)
            combined_text += f"\n--- [엑셀 문서: {file}] ---\n"
            combined_text += df.to_markdown(index=False) + "\n\n"
        except Exception as e:
            st.error(f"⚠️ {file} (엑셀) 읽기 실패: {e}")
            
    pdf_files = glob.glob("*.pdf")
    for file in pdf_files:
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
            
    if not excel_files and not pdf_files:
        return "⚠️ 로딩된 데이터 파일이 없습니다. 시스템 폴더에 .xlsx 또는 .pdf 파일을 업로드해 주십시오."
        
    return combined_text

st.session_state.db_data = load_all_documents()

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

🚨 [절대 준수 규칙: 품목별 관할 법령 동적 매칭 및 연쇄 추론 알고리즘] 🚨
1. 🥛 [품목 기반 법령 자동 분류]: 사용자의 질문에 등장하는 제품이 무엇인지 먼저 파악하십시오.
   - 우유, 가공유, 발효유, 치즈 등 ➡️ 「축산물 위생관리법」 관점 우선 제안
   - 두유, 과채음료, 일반 혼합음료 등 ➡️ 「식품위생법」 관점 우선 제안
   - 품목이 불분명할 경우 두 법령의 관점을 모두 제안
2. ✨[현장 감사 관점]: 사용자의 질문이 표면적인 절차 위반이라도, 실제 공장 감사 시 필연적으로 적발되는 파생 위반(예: 성분배합비율 불일치, 생산작업일지 및 원료출납 서류 허위 작성) 관점을 반드시 1개 이상 끌어내어 제안하십시오.
3. [라벨/표시]: 무조건 「식품표시광고법」 관점을 제안하십시오.
4. "제O조" 같은 구체적인 조항 번호는 절대로 지어내지 마십시오.
5. 부연 설명 없이, 숫자 1, 2, 3으로 시작하는 3줄의 텍스트만 출력하십시오.

사용자 질문: {question}
선택된 키워드: {selected_keyword}
"""

# [NEW] 물리적 키워드 차단(Hard Block) 및 일상어 통역 적용
CASE_TEMPLATE = """
당신은 엑셀 및 PDF 원문 추출 전담 AI입니다.
사용자의 질문, [업종], [위반 구역]을 바탕으로 [마스터 통합 데이터베이스]에 '실제로 존재하는' 텍스트(행 또는 조항)만 3~5개 정확히 복사해서 객관식으로 만드십시오.

🚨 [환각 원천 차단 및 감사 관점 매칭 규칙] 🚨
1. 원문 100% 복사: DB에 없는 조항이나 문구를 단 한 글자라도 지어내면 안 됩니다.
2. 🏢 [업종(Business Type) 철벽 물리적 격리]: 실무자가 지정한 [{business_type}]을 철저히 확인하십시오.
   - 만약 '일반 식품'이 포함되어 있다면, 데이터베이스에서 '도축, 집유, 식육, 식용란' 단어가 들어간 조항은 무조건 제외하십시오.
   - 만약 '축산물'이 포함되어 있다면, 일반 식품위생법 조항을 무조건 제외하십시오.
3. 🧠 [일상어 ↔ 법률어 자동 통역]: 실무자가 지정한 구역 [{selected_category}]나 질문에 "보건증", "지하수", "라벨 텍스트" 같은 현장 일상어가 입력되었더라도, 그 의미를 파악하여 DB 속 정확한 법률 용어(예: 건강진단, 수질검사, 표시사항)와 매칭하여 스캔하십시오.
4. 억지 매칭 금지: 관련 항목이 도저히 없다면 반드시 "⚠️ DB에서 관련 항목을 찾을 수 없습니다"라고만 출력하십시오.
5. 부연 설명 없이 숫자 1, 2, 3으로 시작하십시오.

[마스터 통합 데이터베이스 (엑셀+PDF)]:
{db_data}

사용자 질문: {question}
선택된 법률 방향: {selected_direction}
강제 지정된 업종: {business_type}
강제 지정된 위반 구역: {selected_category}
"""

# [NEW] 조항 번호 오류 및 처분 수위 믹스 방지
TEMPLATE = """
당신은 연세유업의 데이터베이스 통합 추출 전담 AI입니다.
실무자가 최종 선택한 **[세부 위반 상황]**을 [마스터 통합 데이터베이스]에서 찾아내 리포트를 작성하십시오.

🚨 [최종 리포트 도출 3단계 검증 프로세스] 🚨

▶ **[Pass 1: 원문 교차 추출 및 법령 철벽 격리]**
1. 데이터베이스에서 '{selected_case}'와 100% 일치하는 단일 원문(행) 하나만 타겟으로 잡으십시오. 다른 유사한 행과 내용을 섞으면 절대 안 됩니다.
2. ⚠️ [법령 물리적 격리]: 강제 지정된 업종({business_type})이 '일반 식품'을 포함하면 「축산물 위생관리법」 및 '식육' 관련 조항을 절대 출력하지 마십시오.

▶ **[Pass 1.5: 파생 위반 리스크 팩트체크]**
장부 조작 등 연쇄 적발될 수 있는 DB 내 조항을 찾되, 위 [법령 물리적 격리]를 100% 준수하여 해당 업종에 맞는 법령만 스캔하십시오.

---FINAL_REPORT---

▶ **[Pass 2: 최종 리포트 도출 (출력)]**
가독성을 위해 표와 텍스트를 명확히 분리하여 출력하십시오. <br> 태그는 절대로 사용하지 마십시오.

### 📊 위반 사항 및 행정처분 요약

| 구분 | 상세 검토 내용 |
| :--- | :--- |
| **1. 위반 의심 사항** | (질문 요약 및 위반 행위 팩트 기재) |
| **2. 관련 법령 및 조항** | (반드시 '{selected_case}' 또는 DB 원문에 명시된 '법 제O조제O항' 형태의 공식 조항 번호만 기재. 업종명을 적지 마십시오.) |
| **3. 위반사항 (원문)** | (반드시 '{selected_case}'의 텍스트와 100% 일치하는 DB 원문 전체 복사. 임의 요약 금지) |
| **4. 행정처분 수위** | (반드시 '{selected_case}'에 해당하는 1차, 2차, 3차 처분 수위 팩트만 기재. 엉뚱한 조항의 처분을 섞지 마십시오.) |
| **5. 과태료/과징금** | (DB에서 확인된 액수 팩트만 기재, 없으면 '해당 없음') |

---
### 🚨 [중요] 연쇄 파생 위반 리스크
(단속반 현장 감사 관점: 표면적 위반 외에 연쇄적으로 적발될 수 있는 해당 업종 전용 법령 조항 및 처분 수위를 글머리기호(-)를 사용하여 가독성 있게 나열하십시오. 표 안에 넣지 마십시오.)

---
### 💡 품질관리 및 대응 가이드
(글머리기호(1. 2. 3.)를 사용하여 구체적이고 실행 가능한 대처 방안을 나열하십시오. 표 안에 넣지 마십시오.)

[마스터 통합 데이터베이스]:
{db_data}

---
사용자 질문: {question}
강제 지정된 업종: {business_type}
선택된 위반 구역: {selected_category}
실무자가 찾은 원문(위반 상황): {selected_case}
"""

# --- 💡 UI 구성 ---
user_question = st.text_area("🔍 검토가 필요한 위반 의심 사례, 표시사항 누락 등 구체적인 상황을 입력해 주십시오:", height=100, placeholder="예시: 두유 배합비 동일 등록 건 / 우유 제품명 변경 건 등")

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
    st.info("💡 빠르고 정확한 행정처분 기준 조회를 위해, 대상 업종 및 위반 카테고리를 특정해 주십시오.")
    
    st.markdown("#### ① 대상 업종 선택")
    biz_choices = [
        "🏢 식품제조·가공업 (일반 식품, 두유, 과채음료 등)",
        "🥩 축산물가공업 / 식육포장처리업 (유가공품, 우유, 치즈 등)",
        "🏪 즉석판매제조·가공업 / 식품접객업",
        "🌐 공통 적용 (업종 무관)"
    ]
    selected_biz = st.radio("처분 대상 업종:", biz_choices, key="biz_radio")

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
        custom_cat = st.text_input("💡 엑셀에서 집중적으로 검색할 키워드를 직접 입력해 주십시오 (예: 건강진단, 수질검사, 회수명령 등):")
        if custom_cat:
            selected_cat = custom_cat

    if st.button("▶ 5단계: 조항 및 처분기준 조회", type="secondary"):
        with st.spinner("선택된 업종 및 카테고리에 해당하는 사내 DB(법령/처분기준표)를 검색 중입니다..."):
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0)
            st.session_state.case_options = [opt.strip() for opt in (PromptTemplate.from_template(CASE_TEMPLATE) | llm | StrOutputParser()).invoke({
                "db_data": st.session_state.db_data,
                "question": user_question,
                "selected_direction": selected_dir,
                "business_type": selected_biz,
                "selected_category": selected_cat
            }).split('\n') if opt.strip() and opt[0].isdigit()]
            st.session_state.phase = 5

# 5단계
if st.session_state.phase == 5 and st.session_state.case_options:
    st.markdown("### 📋 5단계: 세부 위반 조항 최종 확인 (DB 원문)")
    st.info("💡 데이터베이스에서 검색된 실제 법령 및 처분 조항입니다. 해당하는 내역을 최종 선택해 주십시오.")
    selected_case = st.radio("해당 위반 조항:", st.session_state.case_options, key="case_radio")

    if st.button("📄 최종 법무 검토 및 행정처분 리포트 생성", type="primary"):
        with st.spinner("법령(PDF) 및 처분기준(Excel) 교차 검증을 통한 최종 리포트를 산출하고 있습니다..."):
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0)
            rag_chain = PromptTemplate.from_template(TEMPLATE) | llm | StrOutputParser()
            
            full_response = rag_chain.invoke({
                "db_data": st.session_state.db_data,
                "question": user_question, 
                "business_type": selected_biz,
                "selected_category": selected_cat,
                "selected_case": selected_case
            })
            
        st.markdown("### 📊 최종 법무 검토 및 행정처분 리포트")
        
        if "---FINAL_REPORT---" in full_response:
            reasoning_part, report_part = full_response.split("---FINAL_REPORT---", 1)
            
            with st.expander("🔍 [참고] AI 교차 검색 로그 및 원문 팩트체크 내역 (클릭하여 펼치기)"):
                st.markdown(reasoning_part.strip())
                
            st.markdown(report_part.strip())
        else:
            st.markdown(full_response)
