import streamlit as st
import os
import glob
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="AI 식품/축산물 표시사항 검토 시스템", page_icon="🥛", layout="wide")
st.title("🥛 연세유업 AI 식품/축산물 법령 검토 시스템 (5단계 Pro)")
st.markdown("""
품질안전부문 실무진을 위한 맞춤형 법률 및 규격 검토 도구입니다.
(👨‍⚖️ **5-Step HITL 탑재**: AI의 헛발질(예: 당알코올을 영양성분으로 착각)을 막기 위해 실무자가 위반 구역을 강제로 지정합니다.)
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
if 'excel_data' not in st.session_state:
    st.session_state.excel_data = ""

# --- 💡 엑셀(.xlsx) DB 통째로 로딩 ---
@st.cache_data
def load_all_excel_data():
    excel_files = glob.glob("*.xlsx")
    if not excel_files:
        return "⚠️ 로딩된 엑셀 데이터 파일이 없습니다. 깃허브에 .xlsx 파일이 있는지 확인해주세요."
    
    combined_text = "==== [연세유업 마스터 법령/처분/과태료 데이터베이스] ====\n\n"
    for file in excel_files:
        try:
            df = pd.read_excel(file)
            combined_text += f"\n--- [문서: {file}] ---\n"
            combined_text += df.to_markdown(index=False) + "\n\n"
        except Exception as e:
            st.error(f"⚠️ {file} 읽기 실패: {e}")
    return combined_text

st.session_state.excel_data = load_all_excel_data()

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

🚨 [절대 준수 규칙] 🚨
1. "제O조" 같은 구체적인 조항 번호는 절대로 지어내지 마십시오.
2. 오직 법률의 이름과 어떤 관점인지 간략한 설명만 적으십시오.
3. 부연 설명 없이, 숫자 1, 2, 3으로 시작하는 3줄의 텍스트만 출력하십시오.

사용자 질문: {question}
선택된 키워드: {selected_keyword}
"""

CASE_TEMPLATE = """
당신은 데이터베이스 쟁점 추출 전문가입니다.
사용자의 질문과 실무자가 강제로 지정한 **[위반 구역(카테고리)]**을 바탕으로, [마스터 데이터베이스]에서 조건에 정확히 일치하는 위반행위 3~5개를 원문 그대로 추출하십시오.

🚨 [절대 준수 규칙] 🚨
1. 실무자가 지정한 [위반 구역]을 최우선으로 따르십시오. 
   - 실무자가 '소비자 안전 주의사항'을 골랐다면, 절대 '영양성분' 관련 엑셀 행(Row)을 추출하지 마십시오.
   - 실무자가 '완전 무표시'를 골랐다면, '전부를 표시하지 않은 경우' 관련 행만 추출하십시오.
2. 당신의 사전 지식을 쓰지 말고, 오직 [마스터 데이터베이스]의 '위반행위' 열 텍스트를 글자 하나 바꾸지 말고 그대로 복사하십시오.
3. 부연 설명 없이 숫자 1, 2, 3으로 시작하는 텍스트만 출력하십시오.

[마스터 데이터베이스]:
{excel_data}

사용자 질문: {question}
선택된 법률 방향: {selected_direction}
강제 지정된 위반 구역: {selected_category}
"""

TEMPLATE = """
당신은 연세유업의 최고 권위 식품/축산물 법령 AI 비서입니다.
사용자의 질문과 가장 중요한 **[세부 위반 상황(의사결정 결과)]**을 완벽히 반영하여, 아래 제공된 [마스터 데이터베이스]에서 '정확히 일치하는 단 하나의 행(Row)'을 찾아내십시오.

🚨 [데이터 검색 최우선 규칙] 🚨
1. 사용자가 선택한 [세부 위반 상황]과 100% 일치하는 처분 기준을 찾아내십시오.
2. 데이터베이스에 명시된 1차, 2차, 3차 처분 및 과태료 액수를 단 하나도 빼놓지 말고 그대로 출력하십시오.

💡 **[최종 출력 포맷: 마크다운 표(Table) 형식]** 💡
결과를 줄글로 쓰지 말고, 반드시 아래의 표 양식으로만 출력하십시오.

| 구분 | 상세 검토 내용 |
| :--- | :--- |
| **1. 위반 의심 사항** | (질문 요약 및 어떤 행위가 위반인지 명확히 기재) |
| **2. 관련 법령 및 조항** | (데이터베이스에서 찾은 정확한 조항 번호) |
| **3. 행정처분 수위** | • **1차 처분:** [내용]<br>• **2차 처분:** [내용]<br>• **3차 처분:** [내용] |
| **4. 과태료 및 과징금** | (데이터베이스에 명시된 과태료/과징금 액수, 없으면 '해당 없음') |
| **5. 품질관리 가이드** | 1. (대처 방안 1)<br>2. (대처 방안 2)<br>3. (대처 방안 3) |

[마스터 데이터베이스]:
{excel_data}

---
사용자 질문: {question}
선택된 위반 구역: {selected_category}
선택된 세부 위반 상황: {selected_case}
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

# 3단계 및 4단계 (UI 병합 노출)
if st.session_state.phase >= 3 and st.session_state.direction_options:
    st.markdown("### 🏛️ 3단계: 적용 법률 방향 선택")
    selected_dir = st.radio("적용할 법률 관점:", st.session_state.direction_options, key="dir_radio")
    
    st.markdown("---")
    st.markdown("### 🗂️ 4단계: 위반 구역(표시 유형) 강제 지정")
    st.info("💡 AI가 당알코올(주의사항)과 영양성분을 헷갈리지 못하게 직접 구역을 고정해주세요.")
    
    # 우리가 직접 만든 고정형 카테고리!
    category_choices = [
        "🛑 [소비자 안전 주의사항] 알레르기 주의문구, 당알코올 경고문 등 안전 목적의 문구 누락",
        "📋 [기본 표시/원재료] 제품명, 원재료명, 알레르기 유발물질(원료 자체) 등 일반 정보 누락",
        "📊 [영양표시] 영양정보표 박스 안의 수치(열량, 당류 등) 누락 및 표기 오류",
        "⚠️ [완전 무표시] 라벨 자체를 아예 부착하지 않은 경우"
    ]
    selected_cat = st.radio("위반 쟁점 분류:", category_choices, key="cat_radio")

    if st.button("🔎 5단계: 세부 위반조건 추출", type="secondary"):
        with st.spinner("지정된 구역 안에서만 정확한 위반 케이스를 추출 중입니다..."):
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0)
            st.session_state.case_options = [opt.strip() for opt in (PromptTemplate.from_template(CASE_TEMPLATE) | llm | StrOutputParser()).invoke({
                "excel_data": st.session_state.excel_data,
                "question": user_question,
                "selected_direction": selected_dir,
                "selected_category": selected_cat
            }).split('\n') if opt.strip() and opt[0].isdigit()]
            st.session_state.phase = 4

# 5단계 (최종 확인 및 리포트)
if st.session_state.phase == 4 and st.session_state.case_options:
    st.markdown("### 📋 5단계: 세부 위반 상황 선택 (엑셀 원문 확인)")
    st.info("💡 지정하신 구역에 해당하는 실제 엑셀 처분 기준들입니다.")
    selected_case = st.radio("정확한 위반 상황:", st.session_state.case_options, key="case_radio")

    if st.button("🚀 최종 리포트 생성", type="primary"):
        with st.status("최종 마스터 리포트 작성 중...", expanded=True) as status:
            llm_stream = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0, streaming=True)
            rag_chain = PromptTemplate.from_template(TEMPLATE) | llm_stream | StrOutputParser()
            status.update(label="✅ 분석 완료. 리포트를 출력합니다.", state="complete")
            
        st.markdown("### 📊 최종 분석 결과 리포트")
        st.write_stream(rag_chain.stream({
            "excel_data": st.session_state.excel_data,
            "question": user_question, 
            "selected_category": selected_cat,
            "selected_case": selected_case
        }))
