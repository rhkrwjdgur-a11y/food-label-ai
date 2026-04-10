import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 웹페이지 기본 설정 ---
st.set_page_config(page_title="AI 식품 표시사항 검토 시스템", page_icon="🥛", layout="wide")
st.title("🥛 AI 기반 식품 표시사항 및 행정처분 검토 시스템")
st.markdown("""
품질관리 부서를 위한 표시사항 검토 및 위반 사례 분석 도구입니다. 
좌측 사이드바에 OpenAI API 키와 관련 법률(PDF) 및 실무 FAQ(Excel)를 업로드한 후 질문해주세요.
(예: 환원유를 사용한 가공유 전면 표시사항 누락 시 행정처분 기준 등)
""")

# --- 사이드바 설정 (API 키 및 파일 업로드) ---
with st.sidebar:
    st.header("⚙️ 시스템 설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    
    st.subheader("📄 참조 문서 업로드")
    uploaded_files = st.file_uploader(
        "법률 고시(PDF) 및 FAQ(Excel) 파일 업로드", 
        type=["pdf", "xlsx", "xls"], 
        accept_multiple_files=True
    )

# --- 핵심 RAG 분석 로직 (축약 없음) ---
def process_documents_and_analyze(uploaded_files, user_query, api_key):
    # 환경변수에 API 키 임시 저장
    os.environ["OPENAI_API_KEY"] = api_key
    
    documents = []
    
    # 1. 임시 디렉토리를 생성하여 업로드된 파일을 로컬에 임시 저장 후 로드
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Streamlit의 업로드된 파일 객체를 임시 파일로 쓰기
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # 확장자에 따른 로더 선택
            if temp_file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(temp_file_path)
                documents.extend(loader.load())
            elif temp_file_path.lower().endswith(('.xls', '.xlsx')):
                loader = UnstructuredExcelLoader(temp_file_path)
                documents.extend(loader.load())

    if not documents:
        return "오류: 문서를 정상적으로 읽어오지 못했습니다."

    # 2. 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(documents)

    # 3. 임베딩 및 벡터 스토어 생성
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    # 4. 검색기 설정
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # 5. 시스템 프롬프트 설정 (환각 방지 및 5단계 출력 구조 강제)
    template = """
    당신은 대한민국 식품위생법, 식품 등의 표시·광고에 관한 법률, 농수산물의 원산지 표시 등에 관한 법률을 전문으로 분석하는 AI 법률 검토 보조 시스템입니다.
    사용자가 질문과 함께 제공한 [참조 법률 문서 및 FAQ]만을 바탕으로 답변을 작성해야 합니다. 참조 문서에 명시되지 않은 처분 기준이나 내용을 임의로 생성(Hallucination)하여 답변하는 것을 엄격히 금지합니다. 관련 법령이 참조 문서에 없다면 "제공된 문서에서 해당 위반에 대한 처분 기준을 찾을 수 없습니다"라고 답변하십시오.

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

    # 6. LLM 모델 설정
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 7. LCEL 체인 구성
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 8. 분석 실행
    return rag_chain.invoke(user_query)

# --- 사용자 질의 UI ---
user_question = st.text_area("분석할 표시사항 위반 의심 사례나 질문을 구체적으로 입력하세요:", height=150)

if st.button("분석 실행", type="primary"):
    if not openai_api_key:
        st.warning("⚠️ 좌측 사이드바에 OpenAI API Key를 입력해주세요.")
    elif not uploaded_files:
        st.warning("⚠️ 참조할 법률 PDF나 FAQ 엑셀 파일을 업로드해주세요.")
    elif not user_question:
        st.warning("⚠️ 분석할 질문을 입력해주세요.")
    else:
        with st.spinner("AI가 관련 법령과 FAQ를 분석 중입니다... 잠시만 기다려주세요."):
            try:
                # 결과 도출
                result = process_documents_and_analyze(uploaded_files, user_question, openai_api_key)
                st.success("분석이 완료되었습니다.")
                st.markdown("### 📊 분석 결과 리포트")
                st.info(result)
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")