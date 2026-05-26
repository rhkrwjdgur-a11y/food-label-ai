import streamlit as st
import os
import glob
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# PyPDF2 대신 회원님의 환경에 맞춘 최신 pypdf 사용!
try:
    from pypdf import PdfReader
except ImportError:
    st.error("⚠️ pypdf 라이브러리가 설치되지 않았습니다. requirements.txt를 확인해주세요.")

st.set_page_config(page_title="AI 식품/축산물 표시사항 검토 시스템", page_icon="🥛", layout="wide")
st.title("🥛 연세유업 AI 식품/축산물 법령 검토 시스템 (Pro 마스터버전)")
st.markdown("""
품질안전부문 실무진을 위한 맞춤형 법률 및 규격 검토 도구입니다.
(👨‍⚖️ **통합 검증 시스템**: 엑셀+PDF를 동시 분석하며, 구체적 단어가 없을 시 포괄적 조항을 찾고 3단계 자체 팩트체크를 거칩니다.)
""")

try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("⚠️ 설정(Secrets)에 GOOGLE_API_KEY가 등록되지 않았습니다.")
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
@st.cache_data
def load_all_documents():
    combined_text = "==== [연세유업 마스터 통합 데이터베이스 (엑셀+PDF)] ====\n\n"
    
    # 1. 엑셀 파일 읽기
    excel_files = glob.glob("*.xlsx")
    for file in excel_files:
        try:
            df = pd.read_excel(file)
            combined_text += f"\n--- [엑셀 문서: {file}] ---\n"
            combined_text += df.to_markdown(index=False) + "\n\n"
        except Exception as e:
            st.error(f"⚠️ {file} (엑셀) 읽기 실패: {e}")
            
    # 2. PDF 파일 읽기 (pypdf 적용)
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
        return "⚠️ 로딩된 데이터 파일이 없습니다. 앱 폴더에 .xlsx 또는 .pdf 파일을 넣어주세요."
        
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
사용자의 질문과 지정된 [위반 구역]을 바탕으로, [마스터 통합 데이터베이스]에 '실제로 존재하는' 텍스트(엑셀의 행 또는 PDF의 조항 문장)만 3~5개 정확히 복사해서 객관식으로 만드십시오.

🚨 [환각 원천 차단 및 포괄적 매칭 규칙] 🚨
1. 원문 100% 복사: 엑셀이나 PDF에 없는 조항이나 문구를 단 한 글자라도 지어내면 안 됩니다.
2. ✨ [포괄적 위반 항목 최우선 탐색]: 사용자의 질문(예: 당알코올 주의문구, 특정 경고문 누락 등)에 해당하는 구체적인 단어가 DB에 없을 경우, 엉뚱한 항목(예: 알레르기 등)을 가져오지 마십시오! 대신 반드시 DB 원문 중에서 "표시사항 전부 또는 일부 미표시", "그 밖의 표시사항 미표시" 등과 같이 가장 포괄적인 위반행위 텍스트를 1순위로 찾아서 객관식에 포함시키십시오.
3. 억지 매칭 금지: 관련 항목이 도저히 없다면 반드시 "⚠️ DB에서 관련 항목을 찾을 수 없습니다"라고만 출력하십시오.
4. 실무자 지정 구역 최우선: 4단계에서 지정한 [위반 구역(카테고리)]에 해당하는 범위 안에서만 찾으십시오.
5. 부연 설명 없이 숫자 1, 2, 3으로 시작하십시오.

[마스터 통합 데이터베이스 (엑셀+PDF)]:
{db_data}

사용자 질문: {question}
선택된 법률 방향: {selected_direction}
강제 지정된 위반 구역: {selected_category}
"""

TEMPLATE = """
당신은 연세유업의 데이터베이스 통합 추출(엑셀 VLOOKUP + PDF 검색) 전담 AI입니다.
실무자가 5단계에서 최종 선택한 **[세부 위반 상황]**을 [마스터 통합 데이터베이스]에서 찾아내고, 오직 그 줄(Row)이나 문단에 적혀있는 정보만 사용하여 리포트를 작성하십시오.

🚨 [최종 리포트 도출 3단계 검증 프로세스] 🚨
당신의 뇌피셜(소설)을 막기 위해, 리포트 표를 그리기 전에 반드시 아래의 'Pass 1'과 'Pass 1.5'의 사고 과정을 화면에 먼저 출력하십시오.

▶ **[Pass 1: 원문 추출 (검색)]**
데이터베이스(엑셀 또는 PDF)에서 '{selected_case}'와 가장 잘 일치하는 행(Row)이나 법령 문단을 찾아, 그 텍스트 전체를 복사하여 적으십시오.

▶ **[Pass 1.5: 데이터 자체 검증 (팩트 체크)]**
Pass 1에서 찾은 원문 텍스트 안에 '관련 법령/조항 번호', '처분 수위(1,2,3차)', '과태료'가 구체적으로 적혀있는지 분석하십시오. (원문에 숫자가 적혀있지 않으면 무조건 '내용 없음'으로 판정하고 절대 지어내지 마십시오.)

▶ **[Pass 2: 최종 리포트 도출 (출력)]**
검증된 '팩트'만을 사용하여 아래 마크다운 표를 완성하십시오. (표 안에는 데이터베이스 원문에 없는 말을 절대 추가하지 마십시오)

💡 **[최종 출력 포맷]** 💡
(반드시 Pass 1과 Pass 1.5를 텍스트로 보여준 뒤, 아래 표를 그리십시오.)

| 구분 | 상세 검토 내용 |
| :--- | :--- |
| **1. 위반 의심 사항** | (질문 요약 및 어떤 행위가 위반인지 명확히 기재) |
| **2. 관련 법령 및 조항** | (Pass 1.5에서 확인된 조항 번호, 없으면 '데이터베이스 표기 없음') |
| **3. 행정처분 수위** | • **1차 처분:** [내용]<br>• **2차 처분:** [내용]<br>• **3차 처분:** [내용] (※ 없으면 '해당 없음') |
| **4. 과태료 및 과징금** | (Pass 1.5에서 확인된 액수, 없으면 '해당 없음') |
| **5. 품질관리 가이드** | 1. (위반 사항을 예방하기 위한 실무 대처 방안 1)<br>2. (실무 대처 방안 2)<br>3. (실무 대처 방안 3) |

[마스터 통합 데이터베이스]:
{db_data}

---
사용자 질문: {question}
선택된 위반 구역: {selected_category}
실무자가 찾은 원문(위반 상황): {selected_case}
"""

# --- 💡 UI 구성 ---
user_question = st.text_area("사례나 분석 데이터를 편하게 입력하세요:", height=100)

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🔍 1단계: 핵심 단어 분석", type="primary"):
        if user_question.strip():
            with st.spinner("분석 중..."):
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0.2)
                st.session_state.keyword_options = [opt.strip() for opt in (PromptTemplate.from_template(KEYWORD_TEMPLATE) | llm | StrOutputParser()).invoke({"question": user_question}).split('\n') if opt.strip() and opt[0].isdigit()]
                st.session_state.phase = 2

# 2단계
if st.session_state.phase >= 2 and st.session_state.keyword_options:
    st.markdown("### 🎯 2단계: 법률 키워드(단어) 선택")
    selected_kw = st.radio("적용할 핵심 키워드:", st.session_state.keyword_options, key="kw_radio")
    
    if st.button("⚖️ 3단계: 법률 쟁점 분석", type="secondary"):
        with st.spinner("분석 중..."):
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0.2)
            st.session_state.direction_options = [opt.strip() for opt in (PromptTemplate.from_template(DIRECTION_TEMPLATE) | llm | StrOutputParser()).invoke({"question": user_question, "selected_keyword": selected_kw}).split('\n') if opt.strip() and opt[0].isdigit()]
            st.session_state.phase = 3

# 3단계
if st.session_state.phase >= 3 and st.session_state.direction_options:
    st.markdown("### 🏛️ 3단계: 적용 법률 방향 선택")
    selected_dir = st.radio("적용할 법률 관점:", st.session_state.direction_options, key="dir_radio")
    
    if st.button("➡️ 4단계: 위반 구역 지정하기", type="secondary"):
        st.session_state.phase = 4

# 4단계
if st.session_state.phase >= 4:
    st.markdown("---")
    st.markdown("### 🗂️ 4단계: 위반 구역(표시 유형) 강제 지정")
    st.info("💡 AI가 엉뚱한 구역(예: 영양성분)을 뒤지지 않도록 실무자가 직접 카테고리를 고정해주세요.")
    
    category_choices = [
        "🛑 [소비자 안전 주의사항] 알레르기 주의문구, 당알코올 경고문 등 안전 목적의 문구 누락",
        "📋 [기본 표시/원재료] 제품명, 원재료명, 알레르기 유발물질(원료 자체) 등 일반 정보 누락",
        "📊 [영양표시] 영양정보표 박스 안의 수치(열량, 당류 등) 누락 및 표기 오류",
        "⚠️ [완전 무표시] 라벨 자체를 아예 부착하지 않은 경우",
        "기타 표시기준 위반"
    ]
    selected_cat = st.radio("위반 쟁점 분류:", category_choices, key="cat_radio")

    if st.button("🔎 5단계: 세부 위반조건 추출", type="secondary"):
        with st.spinner("엑셀과 PDF 문서를 동시에 스캔하여 원문을 추출 중입니다..."):
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0)
            st.session_state.case_options = [opt.strip() for opt in (PromptTemplate.from_template(CASE_TEMPLATE) | llm | StrOutputParser()).invoke({
                "db_data": st.session_state.db_data,
                "question": user_question,
                "selected_direction": selected_dir,
                "selected_category": selected_cat
            }).split('\n') if opt.strip() and opt[0].isdigit()]
            st.session_state.phase = 5

# 5단계
if st.session_state.phase == 5 and st.session_state.case_options:
    st.markdown("### 📋 5단계: 세부 위반 상황 선택 (엑셀/PDF 원문 확인)")
    st.info("💡 엑셀 처분기준표 또는 PDF 문서에서 찾아낸 원문입니다.")
    selected_case = st.radio("정확한 위반 상황:", st.session_state.case_options, key="case_radio")

    if st.button("🚀 최종 리포트 생성 (Pass 1/1.5/2 자기검증)", type="primary"):
        with st.status("엑셀/PDF 원본 추출 및 3단계 자체 검증 중...", expanded=True) as status:
            llm_stream = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0, streaming=True)
            rag_chain = PromptTemplate.from_template(TEMPLATE) | llm_stream | StrOutputParser()
            status.update(label="✅ 통합 검증 완료. 리포트를 출력합니다.", state="complete")
            
        st.markdown("### 📊 최종 분석 결과 리포트")
        st.write_stream(rag_chain.stream({
            "db_data": st.session_state.db_data,
            "question": user_question, 
            "selected_category": selected_cat,
            "selected_case": selected_case
        }))
