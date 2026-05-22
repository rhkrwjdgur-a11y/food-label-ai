import streamlit as st
import os
import glob
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="AI 식품/축산물 표시사항 검토 시스템", page_icon="🥛", layout="wide")
st.title("🥛 연세유업 AI 식품/축산물 법령 및 행정처분 검토 시스템 (Final)")
st.markdown("""
품질안전부문 실무진을 위한 맞춤형 법률 및 규격 검토 도구입니다.
(👨‍⚖️ **Multi-step HITL 탑재**: [단어 추상화]와 [법률 쟁점]을 실무자가 직접 이중 선택하여 오답을 원천 차단합니다.)
""")

try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("⚠️ 설정(Secrets)에 GOOGLE_API_KEY가 등록되지 않았습니다.")
    st.stop()

# --- 💡 세션 상태(Session State) 초기화 ---
if 'phase' not in st.session_state:
    st.session_state.phase = 1
if 'keyword_options' not in st.session_state:
    st.session_state.keyword_options = []
if 'direction_options' not in st.session_state:
    st.session_state.direction_options = []
if 'excel_data' not in st.session_state:
    st.session_state.excel_data = ""

# --- 💡 엑셀(CSV) DB 통째로 로딩 ---
@st.cache_data
def load_all_csv_data():
    csv_files = glob.glob("*.csv")
    if not csv_files:
        return "⚠️ 로딩된 CSV 데이터 파일이 없습니다. 앱 폴더에 변환된 csv 파일들을 넣어주세요."
    
    combined_text = "==== [연세유업 마스터 법령/처분/과태료 데이터베이스] ====\n\n"
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            combined_text += f"\n--- [문서: {file}] ---\n"
            combined_text += df.to_markdown(index=False) + "\n\n"
        except Exception as e:
            st.error(f"⚠️ {file} 읽기 실패: {e}")
    return combined_text

st.session_state.excel_data = load_all_csv_data()

# --- 💡 프롬프트 정의 ---
KEYWORD_TEMPLATE = """
당신은 현장 용어를 법률 용어로 번역하는 AI입니다.
사용자의 질문에서 고유명사를 제거하고, 법전에서 검색될 만한 '3가지 다른 뉘앙스의 법률 키워드(행위 본질)' 옵션을 제안하십시오.
절대 부연 설명 없이, 숫자 1, 2, 3으로 시작하는 3줄의 텍스트만 출력하십시오.

사용자 질문: {question}
"""

DIRECTION_TEMPLATE = """
당신은 수석 법무 검토관입니다.
사용자의 원본 질문과 선택된 '법률 키워드'를 결합하여, 이를 처벌할 수 있는 '3가지 법률 적용 방향(관점)'을 제안하십시오.

🚨 [특별법 우선 및 품목별 엄격 분리 원칙] 🚨
1. 질문에 '유기농', '친환경', '인증마크' 관련 내용이 있다면: 오직 「친환경농어업법(유기식품법)」 관점만 적용 (일반 식품표시법 적용 금지)
2. 질문의 품목이 우유 등 유제품이면: 주로 「축산물 위생관리법」 적용
3. 질문의 품목이 두유 등 일반 식품이면: 「식품위생법」 또는 「식품표시광고법」 적용

절대 부연 설명 없이, 숫자 1, 2, 3으로 시작하는 3줄의 텍스트만 출력하십시오.

사용자 질문: {question}
선택된 키워드: {selected_keyword}
"""

TEMPLATE = """
당신은 연세유업의 최고 권위 식품/축산물 법령 AI 비서입니다.
사용자의 질문, 선택한 [키워드], [법률 방향]을 반영하여, 아래 제공된 [마스터 데이터베이스(CSV 결합본)]에서 '정확히 일치하는 위반 행위와 처분 결과'를 찾아내십시오.

🚨 [데이터 검색 최우선 규칙] 🚨
1. 절대로 당신의 사전 지식으로 소설을 쓰지 마십시오. 오직 아래 제공된 [마스터 데이터베이스] 안에서만 찾으십시오.
2. 유기농 마크/인증 위반은 반드시 유기농 관련 표에서만 찾고, 다른 일반 식품표시법 표는 무시하십시오.
3. 데이터베이스에 명시된 1차, 2차, 3차 처분 및 과태료 액수를 단 하나도 빼놓지 말고 그대로 출력하십시오.

💡 **[최종 출력 포맷: 마크다운 표(Table) 형식]** 💡
결과를 주저리주저리 줄글로 쓰지 마십시오. 반드시 아래의 표 양식으로만 한눈에 들어오게 출력하십시오.

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
선택된 키워드: {selected_keyword}
선택된 법률 방향: {selected_direction}
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

if st.session_state.phase >= 2 and st.session_state.keyword_options:
    st.markdown("### 🎯 2단계: 법률 키워드(단어) 선택")
    selected_kw = st.radio("적용할 핵심 키워드:", st.session_state.keyword_options, key="kw_radio")
    
    if st.button("⚖️ 3단계: 법률 쟁점 분석", type="secondary"):
        with st.spinner("분석 중..."):
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0.2)
            st.session_state.direction_options = [opt.strip() for opt in (PromptTemplate.from_template(DIRECTION_TEMPLATE) | llm | StrOutputParser()).invoke({"question": user_question, "selected_keyword": selected_kw}).split('\n') if opt.strip() and opt[0].isdigit()]
            st.session_state.phase = 3

if st.session_state.phase == 3 and st.session_state.direction_options:
    st.markdown("### 🏛️ 4단계: 적용 법률 방향 선택")
    selected_dir = st.radio("적용할 법률 관점:", st.session_state.direction_options, key="dir_radio")
    
    if st.button("🚀 최종 리포트 생성", type="primary"):
        with st.status("마스터 데이터베이스 분석 및 리포트 작성 중...", expanded=True) as status:
            llm_stream = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0, streaming=True)
            rag_chain = PromptTemplate.from_template(TEMPLATE) | llm_stream | StrOutputParser()
            status.update(label="✅ 분석 완료. 리포트를 출력합니다.", state="complete")
            
        st.markdown("### 📊 최종 분석 결과 리포트")
        st.write_stream(rag_chain.stream({
            "excel_data": st.session_state.excel_data,
            "question": user_question, 
            "selected_keyword": selected_kw,
            "selected_direction": selected_dir
        }))
