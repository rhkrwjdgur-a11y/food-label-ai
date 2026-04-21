import streamlit as st
import os
import glob
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # 💡 구글 대신 로컬 초고속 임베딩 사용
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 웹페이지 기본 설정 ---
st.set_page_config(page_title="AI 식품 표시사항 검토 시스템", page_icon="🥛", layout="wide")
st.title("🥛 AI 기반 식품 표시사항 및 행정처분 검토 시스템")
st.markdown("""
품질관리 부서를 위한 표시사항 검토 및 위반 사례 분석 도구입니다. 
(초고속 로컬 임베딩 시스템이 적용되어 대기 시간 없이 즉시 분석합니다.)
""")

# --- API Key 설정 ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("⚠️ 설정(Secrets)에 GOOGLE_API_KEY가 등록되지 않았습니다.")
    st.stop()

# --- 문서 목록화 ---
pre_uploaded_files = glob.glob("*.pdf") + glob.glob("*.xlsx") + glob.glob("*.xls") + glob.glob("*.txt")
DB_PATH = "faiss_index_db_local"  # 충돌 방지를 위해 DB 폴더명 변경

with st.sidebar:
    st.header("📚 AI 학습 데이터 현황")
    st.success(f"총 {len(pre_uploaded_files)}개의 규정 및 핵심 요약 문서가 감지되었습니다.")

# --- 핵심 RAG 로직 (구글 404/429 에러 원천 차단) ---
@st.cache_resource(show_spinner=False)
def load_and_index_documents(_file_list):
    # 💡 [핵심] 구글 API를 쓰지 않고, 가장 빠르고 가벼운 무료 로컬 다국어 모델 사용
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(DB_PATH):
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    documents = []
    progress_bar = st.progress(0, text="🧠 AI 뇌(DB) 최초 생성 중... (파일 정독)")

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

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    ).split_documents(documents)

    if not documents or len(splits) == 0:
        progress_bar.empty()
        return None

    progress_bar.progress(1.0, text=f"✅ 파일 정독 완료! 총 {len(splits)}개의 조각을 초고속 변환 중입니다...")

    # 💡 [핵심] 이제 time.sleep 따위는 필요 없습니다! 
    # 구글 눈치를 안 봐도 되므로 한방에 초고속으로 벡터 DB를 만듭니다.
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(DB_PATH)
    
    progress_bar.empty()
    return vectorstore

# --- 템플릿 정의 ---
TEMPLATE = """
당신은 대한민국 식품위생법, 식품 등의 표시·광고에 관한 법률, 농수산물의 원산지 표시 등에 관한 법률을 전문으로 분석하는 AI 법률 검토 보조 시스템입니다.
사용자가 질문과 함께 제공한 [참조 법률 문서 및 FAQ]만을 바탕으로 답변을 작성해야 합니다. 참조 문서에 명시되지 않은 처분 기준이나 내용을 임의로 생성(Hallucination)하여 답변하는 것을 엄격히 금지합니다.

[특수 검토 규칙]: 
소화가 잘되는 우유 등 특수 유제품의 무지유고형분(SNF) 규격 판정 시, 반드시 Brix 측정값에서 지방값을 뺀 수치를 SNF 값으로 잡아서 계산하십시오.

분석 결과는 다음 구조를 준수하십시오.
1. 위반 의심 사항:
2. 관련 법령, 조항 및 참조 FAQ:
3. 행정처분:
4. 과징금 및 벌칙금 (형사처분):
5. 검토 의견:

[참조 법률 문서 및 FAQ]:
{context}

사용자 질문:
{question}
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- 실행 UI ---
user_question = st.text_area("분석할 표시사항 위반 의심 사례나 질문을 구체적으로 입력하세요:", height=150)

if st.button("분석 실행", type="primary"):
    if not pre_uploaded_files:
        st.warning("⚠️ 서버에 학습할 문서가 없습니다.")
    elif not user_question:
        st.warning("⚠️ 질문을 입력해주세요.")
    else:
        with st.status("📂 문서 준비 중...", expanded=False) as status:
            # 이제 api_key를 넘기지 않아도 내부에서 알아서 DB를 만듭니다!
            vector_db = load_and_index_documents(tuple(pre_uploaded_files))
            status.update(label="✅ 준비 완료", state="complete")

        if vector_db:
            st.markdown("### 📊 분석 결과 리포트")
            retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            prompt = PromptTemplate.from_template(TEMPLATE)

            # 💡 답변은 여전히 똑똑한 구글 Gemini를 사용합니다. (이건 에러가 안 납니다!)
            os.environ["GOOGLE_API_KEY"] = google_api_key
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0,
                streaming=True
            )

            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            st.write_stream(rag_chain.stream(user_question))
