import streamlit as st
import os
import glob
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 웹페이지 기본 설정 ---
st.set_page_config(page_title="AI 식품 표시사항 검토 시스템", page_icon="🥛", layout="wide")
st.title("🥛 연세유업 AI 식품 표시사항 및 행정처분 검토 시스템")
st.markdown("""
품질안전부문 실무진을 위한 맞춤형 법률 및 규격 검토 도구입니다.
(에이전트 RAG 다중 홉 추론 엔진 탑재: 법령과 별표를 스스로 교차 검증합니다.)
""")

# --- API Key 설정 ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("⚠️ 설정(Secrets)에 GOOGLE_API_KEY가 등록되지 않았습니다.")
    st.stop()

# --- 문서 목록화 및 DB 경로 설정 (Agentic RAG 전용) ---
pre_uploaded_files = glob.glob("*.pdf") + glob.glob("*.xlsx") + glob.glob("*.xls") + glob.glob("*.txt")
DB_PATH = "faiss_index_db_agentic_v1" 

# --- 핵심 RAG 로직 (한국어 특화 로컬 임베딩) ---
@st.cache_resource(show_spinner=False)
def load_and_index_documents(_file_list):
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    
    if os.path.exists(DB_PATH):
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    documents = []
    progress_bar = st.progress(0, text="🧠 AI 한국어 뇌(DB) 최초 생성 중...")

    for i, file_path in enumerate(_file_list):
        progress_bar.progress((i + 1) / len(_file_list), text=f"[{i+1}/{len(_file_list)}] 📄 '{file_path}' 정독 중...")
        try:
            if file_path.lower().endswith('.pdf'):
                documents.extend(PyPDFLoader(file_path).load())
            elif file_path.lower().endswith('.txt'):
                documents.extend(TextLoader(file_path, encoding='utf-8').load())
            elif file_path.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
                documents.append(Document(page_content=df.to_string(index=False), metadata={"source": file_path}))
        except Exception as e:
            st.warning(f"⚠️ {file_path} 로딩 실패: {e}")

    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
    if not documents or len(splits) == 0: return None

    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(DB_PATH)
    progress_bar.empty()
    return vectorstore

# --- 💡 연세유업 품질관리 전용 AI 지침 (에이전트 RAG 최종 병합판) ---
TEMPLATE = """
당신은 연세유업의 최고 권위 식품법령 및 품질관리 AI 비서입니다.
사용자의 질문에 대해 오직 제공된 [참조 문서]만을 바탕으로 답변하십시오. 지어내거나 일반 법률 지식을 사용하지 마십시오.

[현장 품질관리 특수 규칙]:
1. **SNF(무지유고형분) 계산**: '소화가 잘되는 우유' 등 검토 시, [Brix 측정값 - 지방값]을 SNF 수치로 산출하여 법적 기준과 대조하십시오.
2. **물 혼입 판정**: 데이터상 '지방'과 'SNF' 수치가 **동시에 하락**한 경우에만 '물 혼입'으로 판단하고 결과에 별표(*) 표시를 하십시오.
3. **이력추적관리**: 이력추적관리번호와 관련된 질문 시, 번호 자체와 수량(포장 단위) 표시는 명확히 구분되어야 함을 현장 실무자에게 강조하십시오.

💡 **[최종 출력 포맷]** 💡
결과는 어떠한 경우에도 반드시 아래의 5가지 목차 구조를 100% 동일하게 유지하여 작성하십시오. 각 항목의 제목은 굵은 글씨(**)로 유지하십시오.

**1. 위반 의심 사항:** (질문에서 문제가 되는 행위를 1~2줄 요약)
**2. 관련 법령, 조항 및 참조 FAQ:** (참조 문서에서 찾은 근거 명시)
**3. 행정처분:** (참조 문서에서 찾은 처분 기준 명시. 문서에 없으면 '확인 불가')
**4. 과징금 및 벌칙금 (형사처분):** (참조 문서에서 찾은 과징금/과태료 기준 명시. 문서에 없으면 '확인 불가')
**5. 검토 의견 (품질관리 가이드):** (연세유업 실무진이 어떻게 대처해야 하는지 3가지 이내의 글머리 기호(•)로 요약)

[참조 문서]:
{context}

사용자 질문:
{question}
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- UI 및 에이전트 RAG 실행 ---
user_question = st.text_area("사례나 분석 데이터를 입력하세요 (예: 위생복 미착용 시 처벌 기준...):", height=150)

if st.button("분석 실행", type="primary"):
    if not pre_uploaded_files:
        st.warning("⚠️ 학습할 문서가 없습니다. 깃허브에 문서를 업로드해주세요.")
        st.stop()
    elif not user_question.strip():
        st.warning("⚠️ 분석할 질문이나 데이터를 입력해주세요.")
        st.stop()
    else:
        with st.status("📂 Agentic RAG 다중 추론 분석 중...", expanded=True) as status:
            vector_db = load_and_index_documents(tuple(pre_uploaded_files))
            
            if vector_db:
                retriever = vector_db.as_retriever(
                    search_type="mmr", 
                    search_kwargs={"k": 15, "fetch_k": 40} 
                )

                llm_fast = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash", 
                    api_key=google_api_key,
                    temperature=0
                )
                
                llm_stream = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash", 
                    api_key=google_api_key,
                    temperature=0,
                    streaming=True
                )

                # 💡 [Agentic RAG 1단계]: 관련 조항 추론
                status.update(label="🔍 1단계: 관련 법령 조항 추론 중...", state="running")
                docs_pass_1 = retriever.invoke(user_question + " 위반 행위 법령 조항")
                context_pass_1 = format_docs(docs_pass_1)

                extraction_prompt = PromptTemplate.from_template(
                    "사용자 질문: {question}\n\n문서: {context}\n\n위 문서에서 사용자 질문의 위반 행위에 해당하는 정확한 '법령 및 조항 번호(예: 법 제3조, 제49조 등)'를 찾아 해당 조항 번호만 짧게 출력하시오. 찾을 수 없으면 '확인 불가'라고 출력하시오."
                )
                extraction_chain = extraction_prompt | llm_fast | StrOutputParser()
                article_number = extraction_chain.invoke({"question": user_question, "context": context_pass_1})
                st.write(f"✔️ 1단계 추론 완료 (탐지된 조항: {article_number})")

                # 💡 [Agentic RAG 2단계]: 조항 기반 행정처분/과태료 교차 검색
                status.update(label=f"🔍 2단계: '{article_number}' 기반 행정처분/과태료 표 교차 검색 중...", state="running")
                query_pass_2 = f"{article_number} 행정처분 과태료 기준 영업정지 별표"
                docs_pass_2 = retriever.invoke(query_pass_2)
                st.write("✔️ 2단계 검색 완료 (처분 기준표 매칭 및 데이터 확보)")

                # 💡 1단계와 2단계의 문서를 모두 병합하여 중복 제거
                combined_docs = docs_pass_1 + docs_pass_2
                unique_contents = set()
                final_docs = []
                for doc in combined_docs:
                    if doc.page_content not in unique_contents:
                        unique_contents.add(doc.page_content)
                        final_docs.append(doc)

                final_context = "\n\n".join(doc.page_content for doc in final_docs)
                status.update(label="✅ 다중 추론 완료. 최종 리포트 생성 시작", state="complete")

        if vector_db:
            st.markdown("### 📊 분석 결과 리포트")
            
            prompt = PromptTemplate.from_template(TEMPLATE)
            rag_chain = (
                {"context": lambda _: final_context, "question": RunnablePassthrough()}
                | prompt | llm_stream | StrOutputParser()
            )

            st.write_stream(rag_chain.stream(user_question))
