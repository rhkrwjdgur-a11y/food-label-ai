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
(한국어 특화 로컬 임베딩과 최적화된 Gemini 엔진이 탑재되어 즉시 분석합니다.)
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

# --- 핵심 RAG 로직 (한국어 특화 로컬 임베딩) ---
@st.cache_resource(show_spinner=False)
def load_and_index_documents(_file_list):
    # 💡 [핵심] 한국어 법률/실무 데이터에 특화된 최고 성능 오픈소스 사서(임베딩) 적용
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

# --- 💡 연세유업 품질관리 전용 AI 지침 (프롬프트 템플릿 완결판) ---
TEMPLATE = """
당신은 연세유업의 최고 권위 식품법령 및 품질관리 AI 비서입니다.
사용자의 질문에 대해 오직 제공된 [참조 문서]만을 바탕으로 답변하십시오. 문서에 없는 내용을 지어내는 것을 엄격히 금지합니다.

[현장 품질관리 특수 규칙]:
1. **SNF(무지유고형분) 계산**: '소화가 잘되는 우유' 등 검토 시, [Brix 측정값 - 지방값]을 SNF 수치로 산출하여 법적 기준과 대조하십시오.
2. **물 혼입 판정**: 분석 데이터상 '지방'과 'SNF' 수치가 **동시에 하락**한 경우에만 '물 혼입(Water Contamination)'으로 판단하고 결과에 별표(*) 표시를 하십시오.
3. **이력추적관리**: 이력추적관리번호와 관련된 질문 시, 번호 자체와 수량(포장 단위) 표시는 명확히 구분되어야 함을 현장 실무자에게 강조하십시오.

💡 **[최종 출력 포맷]** 💡
분석 결과는 어떠한 경우에도 반드시 아래의 5가지 목차 구조를 100% 동일하게 유지하여 작성하십시오. 각 항목의 제목은 굵은 글씨(**)로 유지하십시오.

**1. 위반 의심 사항:** (사용자 질문에서 문제가 되는 행위, 규격 미달, 표시 누락 사항을 1~2줄로 요약)
**2. 관련 법령, 조항 및 참조 FAQ:** (참조 문서에서 찾은 정확한 근거 조항명, 별표 번호, 또는 FAQ 내용을 명시)
**3. 행정처분:** (해당 위반 시 1차, 2차 등 차수별 행정처분 기준 명시. 문서에 없으면 '확인 불가' 처리)
**4. 과징금 및 벌칙금 (형사처분):** (관련 금전적 제재나 벌칙 규정 요약)
**5. 검토 의견 (품질관리 가이드):** (연세유업 현장 실무진이 어떻게 대처해야 하는지, 올바른 표시 방법이나 주의사항을 3가지 이내의 글머리 기호(•)로 요약하여 실질적인 가이드를 제공할 것)

[참조 문서]:
{context}

사용자 질문:
{question}
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- UI 및 실행 (환각 방지 철통 보안) ---
user_question = st.text_area("사례나 분석 데이터를 입력하세요 (예: 식품이력추적관리 번호란에 선물용 수량 표기...):", height=150)

if st.button("분석 실행", type="primary"):
    if not pre_uploaded_files:
        st.warning("⚠️ 학습할 문서가 없습니다. 깃허브에 문서를 업로드해주세요.")
        st.stop()
    elif not user_question.strip():
        st.warning("⚠️ 분석할 질문이나 데이터를 입력해주세요.")
        st.stop()
    else:
        with st.status("📂 데이터 분석 중...", expanded=False) as status:
            vector_db = load_and_index_documents(tuple(pre_uploaded_files))
            status.update(label="✅ 준비 완료", state="complete")

        if vector_db:
            st.markdown("### 📊 분석 결과 리포트")
            
            # 💡 [핵심] 정답 누락 방지를 위해 탐색 범위를 4개에서 8개로 2배 확장
            retriever = vector_db.as_retriever(search_kwargs={"k": 8})
            prompt = PromptTemplate.from_template(TEMPLATE)

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
