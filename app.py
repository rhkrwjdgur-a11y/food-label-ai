import streamlit as st
import os
import glob
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader, TextLoader
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

# --- 깃허브에 올라간 문서들을 자동으로 스캔하여 목록화 ---
pre_uploaded_files = glob.glob("*.pdf") + glob.glob("*.xlsx") + glob.glob("*.xls") + glob.glob("*.txt")

with st.sidebar:
    st.header("📚 AI 학습 데이터 현황")
    st.success(f"총 {len(pre_uploaded_files)}개의 규정 및 핵심 요약 문서가 시스템 뇌에 탑재되었습니다.")
    with st.expander("탑재된 문서 목록 보기"):
        for f in pre_uploaded_files:
            st.write(f"- {f}")

# --- 핵심 RAG 분석 로직 (구글 Gemini 최신 모델 적용) ---
@st.cache_resource 
def load_and_index_documents(_file_list, api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
    documents = []
    
    # 깃허브에 있는 파일들을 직접 읽어옵니다.
    for file_path in _file_list:
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file_path.lower().endswith(('.xls', '.xlsx')):
            loader = UnstructuredExcelLoader(file_path)
            documents.extend(loader.load())
        elif file_path.lower().endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())

    if not documents:
        return None

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(documents)

    # 임베딩 및 벡터 스토어 생성 (구글 최신 전용 임베딩으로 변경 완료!)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

def analyze_query(vectorstore, user_query):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    template = """
    당신은 대한민국 식품위생법, 식품 등의 표시·광고에 관한 법률, 농수산물의 원산지 표시 등에 관한 법률을 전문으로 분석하는 AI 법률 검토 보조 시스템입니다.
    사용자가 질문과 함께 제공한 [참조 법률 문서 및 FAQ]만을 바탕으로 답변을 작성해야 합니다. 참조 문서에 명시되지 않은 처분 기준이나 내용을 임의로 생성(Hallucination)하여 답변하는 것을 엄격히 금지합니다. 관련 법령이 참조 문서에 없다면 "제공된 문서에서 해당 위반에 대한 처분 기준을 찾을 수 없습니다"라고 답변하십시오.

    [특수 검토 규칙]: 
    소화가 잘되는 우유 등 특수 유제품의 무지유고형분(SNF) 규격 판정 시, 별도의 지시가 없더라도 반드시 Brix 측정값에서 지방값을 뺀 수치를 SNF 값으로 잡아서 계산하고 법적 기준 부합 여부를 검증하십시오.

    분석 결과는 반드시 다음 구조를 준수하여 작성하십시오.

    1. 위반 의심 사항: 사용자의 질문 내용 중 어떤 행위가 법률 위반에 해당하는지 명시
    2. 관련 법령, 조항 및 참조 FAQ: [참조 법률 문서 및 FAQ]에서 도출된 정확한 법령명, 조항 또는 FAQ 원문 명시
    3. 행정처분: 해당 위반에 대한 1차, 2차, 3차 행정처분 내용 (예: 시정명령, 영업정지 O일, 품목제조정지 O일, 영업허가 취소 등)
    4. 과징금 및 벌칙금 (형사처분): 해당 위반에 대한 금전적 제재 및 징역형 등의 벌칙 규정 내용
    5. 검토 의견: 품질관리 측면에서 해당 위반을 방지하기 위한 올바른 표시 방법 및 대응 가이드

    [참조 법률 문서 및 FAQ]:
    {context}

    사용자 질문:
    {question}
    """
    prompt = PromptTemplate.from_template(template)
    
    # LLM (Google Gemini 1.5 Pro 모델 사용)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(user_query)

# --- 사용자 질의 UI ---
user_question = st.text_area("분석할 표시사항 위반 의심 사례나 질문을 구체적으로 입력하세요:", height=150)

if st.button("분석 실행", type="primary"):
    if not pre_uploaded_files:
        st.warning("⚠️ 서버에 학습할 문서가 없습니다. 깃허브에 파일을 업로드해주세요.")
    elif not user_question:
        st.warning("⚠️ 분석할 질문을 입력해주세요.")
    else:
        with st.spinner("구글 AI가 탑재된 법령과 FAQ를 기반으로 분석 중입니다... 잠시만 기다려주세요."):
            try:
                # 1. 문서 학습 (캐싱되어 있으면 즉시 로드)
                vector_db = load_and_index_documents(pre_uploaded_files, google_api_key)
                if vector_db is None:
                    st.error("문서를 학습하는 중 오류가 발생했습니다.")
                else:
                    # 2. 질문 분석 실행
                    result = analyze_query(vector_db, user_question)
                    st.success("분석이 완료되었습니다.")
                    st.markdown("### 📊 분석 결과 리포트")
                    st.info(result)
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
