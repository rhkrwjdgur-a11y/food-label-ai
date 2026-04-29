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

st.set_page_config(page_title="AI 식품 표시사항 검토 시스템", page_icon="🥛", layout="wide")
st.title("🥛 연세유업 AI 식품 표시사항 및 행정처분 검토 시스템")
st.markdown("""
품질안전부문 실무진을 위한 맞춤형 법률 및 규격 검토 도구입니다.
(원본 법령 및 별표를 파괴하지 않고 통째로 읽어 다중 추론하는 'Raw Document RAG' 엔진 탑재)
""")

try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("⚠️ 설정(Secrets)에 GOOGLE_API_KEY가 등록되지 않았습니다.")
    st.stop()

# --- 💡 뇌 구조 초기화: 원본 판독 전용 DB ---
pre_uploaded_files = glob.glob("*.pdf") + glob.glob("*.xlsx") + glob.glob("*.xls") + glob.glob("*.txt")
DB_PATH = "faiss_index_db_raw_v1" 

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
            # 마스터 요약 파일은 제외하고 오직 원본 법령 파일만 학습하도록 필터링 권장
            if file_path.lower().endswith('.pdf'):
                documents.extend(PyPDFLoader(file_path).load())
            elif file_path.lower().endswith('.txt'):
                documents.extend(TextLoader(file_path, encoding='utf-8').load())
            elif file_path.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
                # 엑셀 표의 구조를 보존하기 위해 markdown 형식으로 변환하여 학습
                documents.append(Document(page_content=df.to_markdown(index=False), metadata={"source": file_path}))
        except Exception as e:
            st.warning(f"⚠️ {file_path} 로딩 실패: {e}")

    # 💡 [핵심 기술]: 문서를 1,000글자가 아닌 10,000글자(거대 청크) 단위로 잘라 표(Table) 구조가 박살나는 것을 방지
    splits = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000).split_documents(documents)
    if not documents or len(splits) == 0: return None

    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(DB_PATH)
    progress_bar.empty()
    return vectorstore

TEMPLATE = """
당신은 연세유업의 최고 권위 식품법령 및 품질관리 AI 비서입니다.
사용자의 질문에 대해 오직 제공된 [참조 문서]의 **법령 원문과 행정처분/과태료 기준표(별표)**를 직접 분석하여 답변하십시오.

💡 **[원본 교차 검증 추론 규칙]** 💡
당신에게는 쪼개지지 않은 거대한 원본 문서가 제공됩니다. 다음 순서로 원본을 파헤치십시오.
1. [참조 문서]에서 사용자의 질문(위반 행위)이 몇 조 몇 항 위반인지 법령 본문을 찾으십시오.
2. 찾아낸 조항 번호(예: 제3조, 제49조 등)를 바탕으로, 함께 제공된 [참조 문서] 내의 '행정처분 기준표(별표)' 또는 '과태료 부과기준표'에서 해당 조항에 부과되는 처분 수위(1차, 2차 등)를 정확히 추적하여 추출하십시오.

💡 **[최종 출력 포맷]** 💡
**1. 위반 의심 사항:** (질문에서 문제가 되는 행위를 1~2줄 요약)
**2. 관련 법령, 조항 및 참조 FAQ:** (원본 문서에서 찾은 정확한 조항 번호 명시)
**3. 행정처분:** (별표 기준표에서 교차 검증으로 찾아낸 영업정지 등 처분 기준)
**4. 과징금 및 벌칙금 (형사처분):** (별표 기준표에서 찾아낸 과태료 등 기준)
**5. 검토 의견 (품질관리 가이드):** (현장 대처 방안 3가지 요약)

[참조 문서]:
{context}

사용자 질문:
{question}
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

user_question = st.text_area("사례나 분석 데이터를 입력하세요 (예: 원액 한글 라벨 미표시 시 처벌...):", height=150)

if st.button("분석 실행", type="primary"):
    if not pre_uploaded_files:
        st.warning("⚠️ 학습할 문서가 없습니다.")
        st.stop()
    elif not user_question.strip():
        st.warning("⚠️ 질문을 입력해주세요.")
        st.stop()
    else:
        with st.status("📂 원본 법령 교차 추론 중...", expanded=True) as status:
            vector_db = load_and_index_documents(tuple(pre_uploaded_files))
            
            if vector_db:
                # 💡 [핵심 기술]: Gemini의 거대한 뇌 용량을 믿고, 한 번에 80,000글자 이상을 가져와서 읽히도록 세팅
                retriever = vector_db.as_retriever(
                    search_type="mmr", 
                    search_kwargs={"k": 8, "fetch_k": 30} 
                )

                llm_fast = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0)
                llm_stream = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0, streaming=True)

                status.update(label="🔍 1단계: 원본 법령 조항 탐색 중...", state="running")
                docs_pass_1 = retriever.invoke(user_question + " 위반 행위 조항")
                
                # 조항을 찾아내는 미니 에이전트
                extraction_chain = PromptTemplate.from_template("질문: {question}\n문서: {context}\n위반에 해당하는 정확한 '법령 조항 번호(예: 제3조)'만 추출. 없으면 '확인 불가'") | llm_fast | StrOutputParser()
                article_number = extraction_chain.invoke({"question": user_question, "context": format_docs(docs_pass_1)})
                st.write(f"✔️ 1단계: 위반 조항 탐지 완료 ({article_number})")

                status.update(label=f"🔍 2단계: {article_number} 기반 원본 [별표] 처분표 추적 중...", state="running")
                docs_pass_2 = retriever.invoke(f"{article_number} 별표 행정처분 과태료 기준표")
                st.write("✔️ 2단계: 처분 기준표 데이터 확보 완료")

                # 문서 중복 제거 및 병합
                combined_docs = docs_pass_1 + docs_pass_2
                unique_contents = {doc.page_content: doc for doc in combined_docs}.values()
                final_context = "\n\n".join(doc.page_content for doc in unique_contents)
                status.update(label="✅ 추론 완료. 최종 리포트 작성", state="complete")

        if vector_db:
            st.markdown("### 📊 분석 결과 리포트")
            rag_chain = (
                {"context": lambda _: final_context, "question": RunnablePassthrough()}
                | PromptTemplate.from_template(TEMPLATE) | llm_stream | StrOutputParser()
            )
            st.write_stream(rag_chain.stream(user_question))
