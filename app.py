import streamlit as st
import os
import glob
import time
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 웹페이지 기본 설정 ---
st.set_page_config(page_title="AI 식품 표시사항 검토 시스템", page_icon="🥛", layout="wide")
st.title("🥛 AI 기반 식품 표시사항 및 행정처분 검토 시스템")
st.markdown("""
품질관리 부서를 위한 표시사항 검토 및 위반 사례 분석 도구입니다. 
(시스템 내부에 식약처 고시 및 연도별 FAQ 문서가 이미 학습되어 있습니다.)
""")

# --- 스트림릿 비밀 금고에서 골든 키(API Key) 자동으로 불러오기 ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("⚠️ 설정(Secrets)에 GOOGLE_API_KEY가 등록되지 않았습니다. 관리자에게 문의하세요.")
    st.stop()

# --- 깃허브 문서 목록화 ---
pre_uploaded_files = glob.glob("*.pdf") + glob.glob("*.xlsx") + glob.glob("*.xls") + glob.glob("*.txt")
DB_PATH = "faiss_index_db"

with st.sidebar:
    st.header("📚 AI 학습 데이터 현황")
    st.success(f"총 {len(pre_uploaded_files)}개의 규정 및 핵심 요약 문서가 감지되었습니다.")

# --- 지수 백오프(재시도) 로직 적용 함수 ---
def add_with_retry(vectorstore, batch, max_retries=5):
    base_delay = 2
    for attempt in range(max_retries):
        try:
            vectorstore.add_documents(batch)
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"❌ 데이터 전송 최종 실패: {e}")
                raise e
            sleep_time = base_delay * (2 ** attempt)  # 2초, 4초, 8초... 늘려가며 대기
            st.warning(f"⚠️ 구글 서버 지연 감지 (429 에러 회피 중). {sleep_time}초 대기 후 재시도... ({attempt+1}/{max_retries})")
            time.sleep(sleep_time)

# --- 핵심 RAG 분석 로직 ---
@st.cache_resource(show_spinner=False)
def load_and_index_documents(_file_list, api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # 💡 최신 모델 적용 및 404 에러 방지
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    if os.path.exists(DB_PATH):
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    documents = []
    progress_bar = st.progress(0, text="🧠 AI 뇌(DB) 최초 생성 중... (데이터 정독 중)")

    for i, file_path in enumerate(_file_list):
        progress_bar.progress((i + 1) / len(_file_list), text=f"[{i+1}/{len(_file_list)}] 📄 '{file_path}' 정독 중...")
        try:
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file_path.lower().endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
            elif file_path.lower().endswith(('.xls', '.xlsx')):
                # 💡 unstructured 충돌 해결: pandas를 이용한 엑셀 텍스트 추출 로직
                df = pd.read_excel(file_path)
                text_content = df.to_string(index=False)
                documents.append(Document(page_content=text_content, metadata={"source": file_path}))
            else:
                continue
        except Exception as e:
            st.warning(f"⚠️ {file_path} 로딩 실패: {e}")

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    ).split_documents(documents)

    if not documents or len(splits) == 0:
        progress_bar.empty()
        return None

    total_splits = len(splits)
    progress_bar.progress(1.0, text=f"✅ 정독 완료! 총 {total_splits}개의 조각을 구글 서버로 안전하게 전송합니다.")

    # 첫 번째 조각으로 벡터 DB 뼈대 생성
    vectorstore = FAISS.from_documents(documents=[splits[0]], embedding=embeddings)

    # 💡 배치 사이즈 100개 + 지수 백오프 로직 결합 (속도와 안정성 동시 확보)
    batch_size = 100
    for i in range(1, total_splits, batch_size):
        progress_msg = f"🧠 구글 서버 연동 중... ({min(i+batch_size, total_splits)}/{total_splits} 조각 완료)"
        progress_bar.progress(min(1.0, (i + batch_size) / total_splits), text=progress_msg)
        
        batch = splits[i : i + batch_size]
        add_with_retry(vectorstore, batch)  # 재시도 함수 호출
        time.sleep(0.5)  # 기본적으로 0.5초만 빠르게 쉬고, 에러 나면 위 함수에서 길게 쉼

    vectorstore.save_local(DB_PATH)
    progress_bar.empty()
    return vectorstore

# --- 템플릿 및 포맷 정의 ---
TEMPLATE = """
당신은 대한민국 식품위생법, 식품 등의 표시·광고에 관한 법률, 농수산물의 원산지 표시 등에 관한 법률을 전문으로 분석하는 AI 법률 검토 보조 시스템입니다.
사용자가 질문과 함께 제공한 [참조 법률 문서 및 FAQ]만을 바탕으로 답변을 작성해야 합니다. 참조 문서에 명시되지 않은 처분 기준이나 내용을 임의로 생성(Hallucination)하여 답변하는 것을 엄격히 금지합니다. 관련 법령이 참조 문서에 없다면 "제공된 문서에서 해당 위반에 대한 처분 기준을 찾을 수 없습니다"라고 답변하십시오.

[특수 검토 규칙]: 
소화가 잘되는 우유 등 특수 유제품의 무지유고형분(SNF) 규격 판정 시, 별도의 지시가 없더라도 반드시 Brix 측정값에서 지방값을 뺀 수치를 SNF 값으로 잡아서 계산하고 법적 기준 부합 여부를 검증하십시오.

분석 결과는 반드시 다음 구조를 준수하여 작성하십시오.
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
            vector_db = load_and_index_documents(tuple(pre_uploaded_files), google_api_key)
            status.update(label="✅ 준비 완료", state="complete")

        if vector_db:
            st.markdown("### 📊 분석 결과 리포트")
            retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            prompt = PromptTemplate.from_template(TEMPLATE)

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
