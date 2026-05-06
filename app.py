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

st.set_page_config(page_title="AI 식품 표시사항 검토 시스템", page_icon="🥛", layout="wide")
st.title("🥛 연세유업 AI 식품 표시사항 및 행정처분 검토 시스템")
st.markdown("""
품질안전부문 실무진을 위한 맞춤형 법률 및 규격 검토 도구입니다.
(👨‍⚖️ **Human-in-the-loop 탑재**: AI가 제안한 법률 쟁점 중 실무자가 정확한 방향을 선택하여 추론을 지휘합니다.)
""")

try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("⚠️ 설정(Secrets)에 GOOGLE_API_KEY가 등록되지 않았습니다.")
    st.stop()

# --- 💡 세션 상태(Session State) 초기화 ---
# 두 단계로 나누어 처리하기 위해 상태를 기억합니다.
if 'phase' not in st.session_state:
    st.session_state.phase = 1
if 'legal_options' not in st.session_state:
    st.session_state.legal_options = []

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
# 1. 쟁점 제안 프롬프트
OPTIONS_TEMPLATE = """
당신은 수석 식품 법무 검토관입니다.
사용자의 일상적인 구어체 질문을 분석하여, 어떤 법률 관점을 적용하느냐에 따라 달라질 수 있는 '3가지 다른 법률적 해석(방향)'을 제안하십시오.
절대 부연 설명 없이, 숫자 1, 2, 3으로 시작하는 3줄의 텍스트만 출력하십시오.

[출력 예시]
1. [식품위생법 관점 - 물리적 행위] 위생적 취급기준 위반, 분리 보관 미흡, 교차오염 방지 위반
2. [식품표시광고법 관점 - 라벨 표시] 소비자안전 표시사항 위반, 유발물질 혼입 가능성 미표시
3. [식품위생법 관점 - 위해 기준] 기준 및 규격 위반, 오염된 이물 혼입

사용자 질문: {question}
"""

# 2. 최종 답변 작성 프롬프트
TEMPLATE = """
당신은 연세유업의 최고 권위 식품법령 및 품질관리 AI 비서입니다.
사용자의 최초 질문과, 사용자가 직접 선택한 **[분석 방향 키워드]**를 바탕으로 제공된 [참조 문서]를 분석하여 답변하십시오.

🚨 [절대 금지 사항: 처벌 기준 억지로 끼워 맞추기(준용) 금지] 🚨
제공된 별표 기준표에 위반 조항과 일치하는 처벌 내용이 없다면, 절대 유추하지 말고 "확인 불가"라고 명시하십시오.

💡 **[최종 출력 포맷]** 💡
**1. 위반 의심 사항:** (질문과 선택된 키워드 기반 요약)
**2. 관련 법령, 조항 및 참조 FAQ:** (원본 문서에서 찾은 정확한 조항 번호 명시)
**3. 행정처분:** (별표 기준표 내용. 없으면 '확인 불가')
**4. 과징금 및 벌칙금 (형사처분):** (별표 기준표 내용. 없으면 '확인 불가')
**5. 검토 의견 (품질관리 가이드):** (현장 대처 방안 3가지 요약)

사용자 질문: {question}
선택된 분석 방향 키워드: {selected_keyword}

[참조 문서]:
{context}
"""

# --- 💡 UI 구성 ---
user_question = st.text_area("사례나 분석 데이터를 편하게 입력하세요:", height=100)

col1, col2 = st.columns([1, 5])
with col1:
    # 1단계 버튼: 질문을 분석하여 옵션 생성
    if st.button("🔍 1단계: 쟁점 분석", type="primary"):
        if not user_question.strip():
            st.warning("질문을 입력해주세요.")
        else:
            with st.spinner("AI가 법률적 쟁점을 분석 중입니다..."):
                llm_fast = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0.2)
                options_chain = PromptTemplate.from_template(OPTIONS_TEMPLATE) | llm_fast | StrOutputParser()
                raw_options = options_chain.invoke({"question": user_question})
                
                # 텍스트를 리스트로 파싱 (1. 2. 3. 을 기준으로 나눔)
                st.session_state.legal_options = [opt.strip() for opt in raw_options.split('\n') if opt.strip() and opt[0].isdigit()]
                st.session_state.phase = 2

# 2단계 UI: 옵션 선택 및 최종 실행 (phase가 2일 때만 표시됨)
if st.session_state.phase == 2 and st.session_state.legal_options:
    st.markdown("---")
    st.markdown("### 🎯 2단계: 분석 방향 선택")
    st.info("💡 AI가 사용자의 질문에서 다음과 같은 3가지 법률적 쟁점(해석)을 도출했습니다. 가장 의도에 맞는 방향을 선택해 주십시오.")
    
    selected_option = st.radio("적용할 법률 관점 및 키워드:", st.session_state.legal_options)
    
    if st.button("🚀 선택한 방향으로 최종 리포트 생성", type="primary"):
        with st.status("선택된 키워드로 원본 법령 교차 추론 중...", expanded=True) as status:
            vector_db = load_and_index_documents(tuple(pre_uploaded_files))
            if vector_db:
                retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 30})
                llm_fast = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0)
                llm_stream = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0, streaming=True)

                status.update(label=f"🔍 1단계: '{selected_option}' 관련 법령 탐색 중...", state="running")
                # 사용자가 선택한 구체적인 키워드로만 꽉 찬 검색을 수행
                docs_pass_1 = retriever.invoke(selected_option + " 법령 조항")
                
                extraction_prompt = PromptTemplate.from_template(
                    "선택된 분석 방향: {keyword}\n\n문서: {context}\n\n위 방향에 가장 정확히 일치하는 '법령 조항 번호'만 추출. 없으면 '확인 불가'"
                )
                extraction_chain = extraction_prompt | llm_fast | StrOutputParser()
                article_number = extraction_chain.invoke({"keyword": selected_option, "context": format_docs(docs_pass_1)})
                st.write(f"✔️ 위반 조항 탐지 완료 ({article_number})")

                status.update(label=f"🔍 2단계: 처분표(별표) 교차 검증 중...", state="running")
                docs_pass_2 = retriever.invoke(f"{article_number} {selected_option} 행정처분 과태료 기준 별표")
                st.write("✔️ 처분 기준표 데이터 확보 완료")

                combined_docs = docs_pass_1 + docs_pass_2
                unique_contents = {doc.page_content: doc for doc in combined_docs}.values()
                final_context = "\n\n".join(doc.page_content for doc in unique_contents)
                status.update(label="✅ 추론 완료. 최종 리포트 작성", state="complete")

        if vector_db:
            st.markdown("### 📊 분석 결과 리포트")
            rag_chain = (
                PromptTemplate.from_template(TEMPLATE) | llm_stream | StrOutputParser()
            )
            # 스트리밍 출력
            st.write_stream(rag_chain.stream({
                "context": final_context, 
                "question": user_question, 
                "selected_keyword": selected_option
            }))
