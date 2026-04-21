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
st.title("🥛 AI 기반 식품 표시사항 및 행정처분 검토 시스템")
st.markdown("""
연세유업 품질관리 전문가를 위한 맞춤형 법률 및 규격 검토 도구입니다.
(로컬 임베딩과 최적화된 Gemini 1.5 엔진이 탑재되었습니다.)
""")

# --- API Key 설정 ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("⚠️ 설정(Secrets)에 GOOGLE_API_KEY가 등록되지 않았습니다.")
    st.stop()

# --- 문서 목록화 및 저장 경로 ---
pre_uploaded_files = glob.glob("*.pdf") + glob.glob("*.xlsx") + glob.glob("*.xls") + glob.glob("*.txt")
DB_PATH = "faiss_index_db_local" 

# --- 핵심 RAG 로직 (로컬 임베딩) ---
@st.cache_resource(show_spinner=False)
def load_and_index_documents(_file_list):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(DB_PATH):
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    documents = []
    progress_bar = st.progress(0, text="🧠 AI 뇌(DB) 생성 중...")

    for i, file_path in enumerate(_file_list):
        progress_bar.progress((i + 1) / len(_file_list), text=f"[{i+1}/{len(_file_list)}] 📄 '{file_path}' 로딩 중...")
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

# --- 💡 연세유업 품질관리 전용 AI 지침 (Template) ---
TEMPLATE = """
당신은 대한민국 식품법령 및 유가공 품질관리 기준을 분석하는 AI 전문가입니다.
제공된 [참조 문서]를 바탕으로 답변하되, 아래의 **현장 품질관리 특수 규칙**을 최우선으로 적용하십시오.

[현장 품질관리 특수 규칙]:
1. **SNF(무지유고형분) 계산**: '소화가 잘되는 우유(Lactose-free)' 등 유제품의 규격 검토 시, 별도의 언급이 없더라도 반드시 [Brix 측정값 - 지방값]을 SNF 수치로 산출하여 법적 기준과 대조하십시오.
2. **물 혼입 판정**: 분석 데이터상 '지방'과 'SNF' 수치가 **동시에 하락**한 경우에만 '물 혼입(Water Contamination)'으로 판단하고 결과에 별표(*) 표시를 하십시오. 두 수치 중 하나만 떨어진 경우는 물 혼입으로 간주하지 않습니다.

분석 결과는 반드시 다음 구조로 작성하십시오.
1. 위반 의심 사항 (또는 규격 판정 결과):
2. 관련 법령/조항 및 내부 규격:
3. 행정처분 및 조치사항:
4. 검토 의견: (Brix/SNF 계산 결과 및 물 혼입 여부 포함)

[참조 문서]:
{context}

사용자 질문:
{question}
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- UI 및 실행 ---
user_question = st.text_area("사례나 분석 데이터를 입력하세요 (예: 소화가 잘되는 우유 Brix 12, 지방 3.5...):", height=150)

if st.button("분석 실행", type="primary"):
    if not pre_uploaded_files:
        st.warning("⚠️ 학습할 문서가 없습니다.")
    elif not user_question:
        st.warning("⚠️ 질문을 입력해주세요.")
    else:
        with st.status("📂 데이터 분석 중...", expanded=False) as status:
            vector_db = load_and_index_documents(tuple(pre_uploaded_files))
            status.update(label="✅ 준비 완료", state="complete")

        if vector_db:
            st.markdown("### 📊 분석 결과 리포트")
            retriever = vector_db.as_retriever(search_kwargs={"k": 4})
            prompt = PromptTemplate.from_template(TEMPLATE)

            # 💡 [404 에러 종결] 가장 표준적인 모델명 사용
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                api_key=google_api_key,
                temperature=0,
                streaming=True
            )

            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt | llm | StrOutputParser()
            )

            st.write_stream(rag_chain.stream(user_question))
