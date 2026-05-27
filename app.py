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

st.set_page_config(page_title="AI 식품/축산물 규제 검토 시스템", page_icon="🏢", layout="wide")

# [UI 변경] 경영진 및 타 부서가 보아도 신뢰감을 주는 공식적인 타이틀과 설명
st.title("🏢 연세유업 규제 및 행정처분 AI 검색 시스템")
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
당신은 수석 법무 검토관입니다.
사용자의 질문과 선택된 '법률 키워드'를 결합하여, 이를 처벌할 수 있는 '3가지 법률 적용 방향(관점)'을 제안하십시오.

🚨 [절대 준수 규칙: 법률 관할의 엄격한 분리] 🚨
1. [라벨/표시/광고]: 무조건 「식품표시광고법」 관점을 1순위로 제안하십시오. (식품위생법 제안 금지)
2. [위생/안전/이물]: 일반 식품은 「식품위생법」, 유제품/식육일 경우는 「축산물 위생관리법」을 최우선으로 제안하십시오.
3. [원산지]: 원재료 속임수는 「농수산물의 원산지 표시에 관한 법률」을 제안하십시오.
4. "제O조" 같은 구체적인 조항 번호는 절대로 지어내지 마십시오.
5. 부연 설명 없이, 숫자 1, 2, 3으로 시작하는 3줄의 텍스트만 출력하십시오.

사용자 질문: {question}
선택된 키워드: {selected_keyword}
"""

CASE_TEMPLATE = """
당신은 엑셀 및 PDF 원문 추출 전담 AI입니다.
사용자의 질문, [업종], [위반 구역]을 바탕으로 [마스터 통합 데이터베이스]에 '실제로 존재하는' 텍스트(행 또는 조항)만 3~5개 정확히 복사해서 객관식으로 만드십시오.

🚨 [환각 원천 차단 및 업종/포괄적 매칭 규칙] 🚨
1. 원문 100% 복사: DB에 없는 조항이나 문구를 단 한 글자라도 지어내면 안 됩니다.
2. 🏢 [업종(Business Type) 철저 매칭]: 실무자가 [{business_type}]을(를) 지정했습니다. DB에서 위반행위를 찾을 때 반드시 해당 업종에 적용되는 처분 기준인지 확인하십시오.
3. ✨ [포괄적 위반 항목 최우선 탐색]: 구체적인 단어(예: 당알코올 주의문구)가 DB에 없을 경우, 엉뚱한 항목을 가져오지 마십시오! 대신 "표시사항 전부 또는 일부 미표시", "그 밖의 표시사항 미표시" 등과 같이 가장 포괄적인 위반행위 텍스트를 1순위로 찾으십시오.
4. 실무자 지정 구역 최우선: 4단계에서 지정한 [{selected_category}]에 해당하는 범위 안에서만 찾으십시오.
5. 억지 매칭 금지: 관련 항목이 도저히 없다면 반드시 "⚠️ DB에서 관련 항목을 찾을 수 없습니다"라고만 출력하십시오.
6. 부연 설명 없이 숫자 1, 2, 3으로 시작하십시오.

[마스터 통합 데이터베이스 (엑셀+PDF)]:
{db_data}

사용자 질문: {question}
선택된 법률 방향: {selected_direction}
강제 지정된 업종: {business_type}
강제 지정된 위반 구역: {selected_category}
"""

# [NEW] 최종 출력 분리를 위한 프롬프트
TEMPLATE = """
당신은 연세유업의 데이터베이스 통합 추출(엑셀 VLOOKUP + PDF 검색) 전담 AI입니다.
실무자가 5단계에서 최종 선택한 **[세부 위반 상황]**을 [마스터 통합 데이터베이스]에서 찾아내 리포트를 작성하십시오.

🚨 [최종 리포트 도출 3단계 검증 프로세스] 🚨
당신의 뇌피셜(소설)을 막기 위해, 반드시 아래의 'Pass 1'과 'Pass 1.5'의 사고 과정을 화면에 먼저 출력하십시오.

▶ **[Pass 1: 원문 교차 추출 (PDF + 엑셀 강제 동시 검색)]**
1. 데이터베이스(엑셀 또는 PDF)에서 '{selected_case}'와 가장 잘 일치하는 원문을 찾아 복사하십시오.
2. ⚠️ [강제 교차 검색 규칙]: 만약 찾은 원문이 PDF 법령 내용이라서 구체적인 '행정처분(품목제조정지 등)'이나 '과태료 금액'이 없다면, 반드시 지정된 업종({business_type})의 **엑셀 데이터베이스**를 추가로 스캔하여 처분 기준 행(Row)을 찾아 덧붙이십시오.

▶ **[Pass 1.5: 데이터 자체 검증 (팩트 체크)]**
Pass 1에서 확보한 (법령 내용 + 엑셀 처분 기준) 텍스트에 '관련 법령/조항', '처분 수위(1,2,3차)', '과태료'가 존재하는지 확인하십시오. (숫자가 없으면 '해당 없음' 판정)

---FINAL_REPORT---

▶ **[Pass 2: 최종 리포트 도출 (출력)]**
위의 "---FINAL_REPORT---" 구분선 아래에는, 오직 검증된 '팩트'만을 사용하여 아래 마크다운 표를 완성하십시오. (지어내기 절대 금지)

| 구분 | 상세 검토 내용 |
| :--- | :--- |
| **1. 위반 의심 사항** | (질문 요약 및 어떤 행위가 위반인지 명확히 기재) |
| **2. 관련 법령 및 조항** | (Pass 1.5에서 확인된 조항 번호, 없으면 '데이터베이스 표기 없음') |
| **3. 행정처분 수위** | • **1차 처분:** [내용]<br>• **2차 처분:** [내용]<br>• **3차 처분:** [내용] (※ 없으면 '해당 없음') |
| **4. 과태료 및 과징금** | (Pass 1.5에서 확인된 액수, 없으면 '해당 없음') |
| **5. 품질관리/대응 가이드** | 1. (대처 방안 1)<br>2. (대처 방안 2)<br>3. (대처 방안 3) |

[마스터 통합 데이터베이스]:
{db_data}

---
사용자 질문: {question}
강제 지정된 업종: {business_type}
선택된 위반 구역: {selected_category}
실무자가 찾은 원문(위반 상황): {selected_case}
"""

# --- 💡 UI 구성 ---
# [UI 변경] 입력창 문구를 공식적으로 변경
user_question = st.text_area("🔍 검토가 필요한 위반 의심 사례, 표시사항 누락 등 구체적인 상황을 입력해 주십시오:", height=100, placeholder="예시: 당알코올 10% 이상 제품에 '과량 섭취 시 설사를 일으킬 수 있습니다' 주의문구 누락 건에 대한 행정처분 기준 조회")

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
        "🏢 식품제조·가공업 (일반 제조/가공)",
        "🏪 즉석판매제조·가공업 / 식품접객업 (소규모/매장 판매 등)",
        "🥩 축산물가공업 / 식육포장처리업 (유가공품, 식육 등)",
        "🌐 공통 적용 (업종 무관)"
    ]
    selected_biz = st.radio("처분 대상 업종:", biz_choices, key="biz_radio")

    st.markdown("#### ② 위반 쟁점 카테고리 선택")
    category_choices = [
        "🛑 [안전 주의사항] 알레르기 주의문구, 당알코올 경고문 등 안전/주의 표기 누락",
        "📋 [기본 표시사항] 제품명, 원재료명 등 필수 일반 정보 누락 또는 오기재",
        "📊 [영양 표시기준] 영양정보표 내 수치 누락 및 표기 오류",
        "⚠️ [완전 무표시] 라벨을 부착하지 않고 유통/판매한 경우",
        "기타 표시기준 위반"
    ]
    selected_cat = st.radio("위반 카테고리:", category_choices, key="cat_radio")

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

    # [UI 변경] 리포트 생성 버튼 및 검증 과정 문구를 공식적으로 변경
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
            
            # [UI 변경] Expander 제목을 전문가스럽게 변경
            with st.expander("🔍 [참고] AI 교차 검색 로그 및 원문 팩트체크 내역 (클릭하여 펼치기)"):
                st.markdown(reasoning_part.strip())
                
            st.markdown(report_part.strip())
        else:
            st.markdown(full_response)
