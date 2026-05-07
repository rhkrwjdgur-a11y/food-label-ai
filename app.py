import streamlit as st
import os
import glob
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="AI 식품/축산물 표시사항 검토 시스템", page_icon="🥛", layout="wide")
st.title("🥛 연세유업 AI 식품/축산물 법령 및 행정처분 검토 시스템")
st.markdown("""
품질안전부문 실무진을 위한 맞춤형 법률 및 규격 검토 도구입니다.
(👨‍⚖️ **Multi-step HITL 탑재**: [단어 추상화]와 [법률 쟁점]을 실무자가 직접 이중 선택하여 오답을 원천 차단합니다.)
""")

try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("⚠️ 설정(Secrets)에 GOOGLE_API_KEY가 등록되지 않았습니다.")
    st.stop()

# --- 💡 세션 상태(Session State) 3단계 초기화 ---
if 'phase' not in st.session_state:
    st.session_state.phase = 1
if 'keyword_options' not in st.session_state:
    st.session_state.keyword_options = []
if 'direction_options' not in st.session_state:
    st.session_state.direction_options = []

# --- 💡 원본 판독 전용 DB ---
pre_uploaded_files = glob.glob("*.pdf") + glob.glob("*.xlsx") + glob.glob("*.xls")
DB_PATH = "faiss_index_db_raw_v2" 

@st.cache_resource(show_spinner=False)
def load_and_index_documents(_file_list):
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    if os.path.exists(DB_PATH):
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    documents = []
    progress_bar = st.progress(0, text="🧠 원본 문서(법령/표) 구조 보존 학습 중...")
    for i, file_path in enumerate(_file_list):
        progress_bar.progress((i + 1) / len(_file_list), text=f"[{i+1}/{len(_file_list)}] 📄 '{file_path}' 정독 중...")
        try:
            if file_path.lower().endswith('.pdf'):
                documents.extend(PyPDFLoader(file_path).load())
            elif file_path.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
                documents.append(Document(page_content=df.to_markdown(index=False), metadata={"source": file_path}))
        except Exception as e:
            st.warning(f"⚠️ {file_path} 로딩 실패: {e}")

    splits = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000).split_documents(documents)
    if not documents or len(splits) == 0: return None

    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(DB_PATH)
    progress_bar.empty()
    return vectorstore

# --- 💡 프롬프트 정의 ---
# [1단계] 단어 추상화 제안 프롬프트
KEYWORD_TEMPLATE = """
당신은 현장 용어를 법률 용어로 번역하는 AI입니다.
사용자의 질문에서 고유명사(알레르기, 원액두유 등)를 제거하고, 법전에서 검색될 만한 '3가지 다른 뉘앙스의 법률 키워드(행위 본질)' 옵션을 제안하십시오.
절대 부연 설명 없이, 숫자 1, 2, 3으로 시작하는 3줄의 텍스트만 출력하십시오.

[출력 예시]
1. 위생적 취급기준 위반, 교차오염 방지 미흡
2. 무표시 제품 판매, 표시사항 전부 미표시
3. 기준 및 규격 위반, 이물 혼입

사용자 질문: {question}
"""

# [2단계] 법률 방향 제안 프롬프트
DIRECTION_TEMPLATE = """
당신은 수석 법무 검토관입니다.
사용자의 원본 질문과, 사용자가 선택한 '법률 키워드'를 결합하여, 이를 처벌할 수 있는 '3가지 법률 적용 방향(관점)'을 제안하십시오.
연세유업은 일반 식품과 유제품(축산물)을 모두 다루므로 「식품위생법」, 「식품표시광고법」, 「축산물 위생관리법」 등을 폭넓게 고려하십시오.
절대 부연 설명 없이, 숫자 1, 2, 3으로 시작하는 3줄의 텍스트만 출력하십시오.

[출력 예시]
1. [축산물 위생관리법] 축산물 가공기준 위반에 따른 유가공업자 행정처분
2. [식품위생법] 위해식품 등 판매 금지 위반에 따른 과태료/행정처분
3. [식품표시광고법] 표시기준 위반에 따른 시정명령 및 과태료

사용자 질문: {question}
선택된 키워드: {selected_keyword}
"""

# [3단계] 최종 리포트 프롬프트
TEMPLATE = """
당신은 연세유업의 최고 권위 식품/축산물 법령 AI 비서입니다.
사용자의 질문, 사용자가 선택한 [키워드], 그리고 [법률 방향]을 모두 반영하여 [참조 문서]를 분석하십시오.

🚨 [처벌 기준 탐색 규칙 - '조항 번호', '상하위 법령' 및 '업종' 우선 매칭] 🚨
1. 1/2번 항목에서 찾은 '위반 법령 조항 번호'를 별표 기준표(엑셀)에서 찾으십시오.
2. 연세유업은 제조기업이므로 반드시 **'축산물가공업', '유가공업', '식품제조·가공업', '공통' 기준**만 가져오십시오. (식품접객업 등 타 업종 절대 금지)
3. 엑셀 텍스트가 100% 똑같지 않아도, 조항 번호와 업종이 일치하면 처분 수위를 가져오십시오.
4. **상하위 법령 교차 매칭:** 시행규칙이나 별표 번호로 못 찾으면, 상위 법률 조항 번호(예: 제3조, 제4조)로 교차 검색하여 도출하십시오. 이래도 없으면 '확인 불가'를 출력하십시오.

💡 **[최종 출력 포맷]** 💡
**1. 위반 의심 사항:** (질문과 선택된 옵션 기반 요약)
**2. 관련 법령, 조항 및 참조 FAQ:** (원본 문서에서 찾은 정확한 조항 번호)
**3. 행정처분:** (조항/업종 기준 매칭. 과태료 사안이면 '해당 없음')
**4. 과징금 및 벌칙금 (형사처분):** (조항/업종 기준 매칭 과태료. 행정처분 사안이면 '해당 없음')
**5. 검토 의견 (품질관리 가이드):** (현장 대처 방안 3가지)

사용자 질문: {question}
선택된 키워드: {selected_keyword}
선택된 법률 방향: {selected_direction}

[참조 문서]:
{context}
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- 💡 UI 구성 (3단계 Funnel) ---
st.write("---")
user_question = st.text_area("사례나 분석 데이터를 편하게 입력하세요:", height=100)

col1, col2 = st.columns([1, 5])
with col1:
    # 1단계: 키워드 분석
    if st.button("🔍 1단계: 핵심 단어 분석", type="primary"):
        if not user_question.strip():
            st.warning("질문을 입력해주세요.")
        else:
            with st.spinner("질문의 법률적 핵심 단어를 추출 중입니다..."):
                llm_fast = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0.2)
                kw_chain = PromptTemplate.from_template(KEYWORD_TEMPLATE) | llm_fast | StrOutputParser()
                raw_kws = kw_chain.invoke({"question": user_question})
                st.session_state.keyword_options = [opt.strip() for opt in raw_kws.split('\n') if opt.strip() and opt[0].isdigit()]
                st.session_state.phase = 2

# 2단계: 키워드 선택 및 방향 분석
if st.session_state.phase >= 2 and st.session_state.keyword_options:
    st.markdown("### 🎯 2단계: 법률 키워드(단어) 선택")
    st.info("💡 AI가 질문에서 추출한 핵심 키워드입니다. 가장 적절한 단어를 선택해 주세요.")
    selected_kw = st.radio("적용할 핵심 키워드:", st.session_state.keyword_options, key="kw_radio")
    
    if st.button("⚖️ 3단계: 법률 쟁점 분석", type="secondary"):
        with st.spinner("선택된 키워드를 바탕으로 적용할 법률을 분석 중입니다..."):
            llm_fast = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0.2)
            dir_chain = PromptTemplate.from_template(DIRECTION_TEMPLATE) | llm_fast | StrOutputParser()
            raw_dirs = dir_chain.invoke({"question": user_question, "selected_keyword": selected_kw})
            st.session_state.direction_options = [opt.strip() for opt in raw_dirs.split('\n') if opt.strip() and opt[0].isdigit()]
            st.session_state.phase = 3

# 3단계: 방향 선택 및 최종 리포트
if st.session_state.phase == 3 and st.session_state.direction_options:
    st.markdown("### 🏛️ 4단계: 적용 법률 방향 선택")
    st.info("💡 선택하신 단어를 바탕으로 AI가 제안하는 법률 방향입니다. 최종 타겟을 선택해 주십시오.")
    selected_dir = st.radio("적용할 법률 관점:", st.session_state.direction_options, key="dir_radio")
    
    if st.button("🚀 최종 리포트 생성", type="primary"):
        with st.status("선택된 옵션으로 원본 법령 교차 추론 중...", expanded=True) as status:
            vector_db = load_and_index_documents(tuple(pre_uploaded_files))
            if vector_db:
                retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 30})
                llm_fast = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0)
                llm_stream = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0, streaming=True)

                status.update(label=f"🔍 1단계: '{selected_dir[:15]}...' 관련 법령 탐색 중...", state="running")
                # 키워드와 방향을 모두 결합하여 극강의 정확도로 검색
                docs_pass_1 = retriever.invoke(f"{selected_kw} {selected_dir} 법령 조항")
                
                extraction_prompt = PromptTemplate.from_template(
                    "사용자 질문: {question}\n선택 키워드: {kw}\n선택 방향: {dir}\n\n문서: {context}\n\n위 내용에 가장 정확히 일치하는 '법령 조항 번호(예: 제3조)'만 추출. 하위 규정(규칙)인 경우 상위 법률 번호도 함께 추출. 없으면 '확인 불가'"
                )
                extraction_chain = extraction_prompt | llm_fast | StrOutputParser()
                article_number = extraction_chain.invoke({
                    "question": user_question, "kw": selected_kw, "dir": selected_dir, "context": format_docs(docs_pass_1)
                })
                st.write(f"✔️ 위반 조항 탐지 완료 ({article_number})")

                status.update(label=f"🔍 2단계: 처분표(별표) 교차 검증 중...", state="running")
                docs_pass_2 = retriever.invoke(f"{article_number} {selected_dir} 행정처분 과태료 기준 별표")
                st.write("✔️ 처분 기준표 데이터 확보 완료")

                combined_docs = docs_pass_1 + docs_pass_2
                unique_contents = {doc.page_content: doc for doc in combined_docs}.values()
                final_context = "\n\n".join(doc.page_content for doc in unique_contents)
                status.update(label="✅ 추론 완료. 최종 리포트 작성", state="complete")

        if vector_db:
            st.markdown("### 📊 최종 분석 결과 리포트")
            rag_chain = (
                PromptTemplate.from_template(TEMPLATE) | llm_stream | StrOutputParser()
            )
            st.write_stream(rag_chain.stream({
                "context": final_context, 
                "question": user_question, 
                "selected_keyword": selected_kw,
                "selected_direction": selected_dir
            }))
