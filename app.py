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
(🔍 **Query Translator 탑재**: 구어체 질문을 법률 용어로 자동 번역하여 원본 법령을 다중 추론합니다.)
""")

try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("⚠️ 설정(Secrets)에 GOOGLE_API_KEY가 등록되지 않았습니다.")
    st.stop()

# --- 💡 원본 판독 전용 DB (요약본 텍스트 배제) ---
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

    # 거대 청크(10,000자) 유지
    splits = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000).split_documents(documents)
    if not documents or len(splits) == 0: return None

    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(DB_PATH)
    progress_bar.empty()
    return vectorstore

# --- 💡 0단계: 질문 번역기(Query Translator) 프롬프트 ---
TRANSLATOR_TEMPLATE = """
당신은 대한민국 최고의 식품위생법 및 식품표시광고법 전문 변호사입니다.
사용자의 일상적인 구어체 현장 용어를 법전(시행령, 시행규칙, 행정처분 별표)에서 정확히 매칭될 수 있는 '공식 법률 용어'로 번역하십시오.

[번역 예시]
- 일상어: "라벨 안 붙이고 원액 팔았어", "탱크로리 라벨 없음" -> 법률어: "무표시 제품 판매, 표시사항 전부 미표시"
- 일상어: "알레르기 원료 따로 보관 안함" -> 법률어: "위생적 취급기준 위반, 교차오염 방지 분리 보관 미흡"
- 일상어: "위생복 안 입음" -> 법률어: "위생복 미착용, 위생적 취급기준 위반"
- 일상어: "유통기한 지난 거 씀" -> 법률어: "소비기한 경과 제품 사용 및 보관"

사용자 질문: {question}
번역된 법률 키워드 (단어 위주로 3~5개 나열):
"""

# --- 최종 답변 작성 프롬프트 ---
TEMPLATE = """
당신은 연세유업의 최고 권위 식품법령 및 품질관리 AI 비서입니다.
사용자의 질문에 대해 오직 제공된 [참조 문서]의 **법령 원문과 행정처분/과태료 기준표(별표)**를 직접 분석하여 답변하십시오.

💡 **[최종 출력 포맷]** 💡
**1. 위반 의심 사항:** (질문에서 문제가 되는 행위를 1~2줄 요약)
**2. 관련 법령, 조항 및 참조 FAQ:** (원본 문서에서 찾은 정확한 법령명 및 조항 번호 명시)
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

user_question = st.text_area("사례나 분석 데이터를 편하게 입력하세요 (예: 라벨 안 붙이고 원액 팔면 어떻게 돼?):", height=150)

if st.button("분석 실행", type="primary"):
    if not pre_uploaded_files:
        st.warning("⚠️ 학습할 문서(PDF, Excel)가 없습니다.")
        st.stop()
    elif not user_question.strip():
        st.warning("⚠️ 질문을 입력해주세요.")
        st.stop()
    else:
        with st.status("📂 원본 법령 교차 추론 중...", expanded=True) as status:
            vector_db = load_and_index_documents(tuple(pre_uploaded_files))
            
            if vector_db:
                retriever = vector_db.as_retriever(
                    search_type="mmr", 
                    search_kwargs={"k": 8, "fetch_k": 30} 
                )

                llm_fast = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0)
                llm_stream = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key, temperature=0, streaming=True)

                # 💡 0단계: 질문 번역기 실행
                status.update(label="🔍 0단계: 구어체 질문을 법률 용어로 번역 중...", state="running")
                translator_chain = PromptTemplate.from_template(TRANSLATOR_TEMPLATE) | llm_fast | StrOutputParser()
                legal_keywords = translator_chain.invoke({"question": user_question})
                st.write(f"✔️ 번역된 법률 키워드: **{legal_keywords.strip()}**")

                # 💡 1단계: 번역된 키워드로 원본 법령 검색
                status.update(label="🔍 1단계: 번역된 키워드로 관련 법령 탐색 중...", state="running")
                docs_pass_1 = retriever.invoke(legal_keywords + " 위반 행위 조항 식품위생법")
                
                extraction_prompt = PromptTemplate.from_template(
                    "사용자의 상황: {question}\n\n문서: {context}\n\n"
                    "당신은 엄격한 판사입니다. 위 문서에서 사용자 상황의 위반 행위 본질과 정확히 일치하는 '법령 조항 번호(예: 제3조)'를 추출하십시오.\n"
                    "절대 억지로 추출하지 말고, 맥락이 맞지 않으면 '확인 불가'라고 출력하십시오."
                )
                extraction_chain = extraction_prompt | llm_fast | StrOutputParser()
                article_number = extraction_chain.invoke({"question": legal_keywords, "context": format_docs(docs_pass_1)})
                st.write(f"✔️ 1단계: 위반 조항 탐지 완료 ({article_number})")

                # 💡 2단계: 조항 기반 별표 추적
                status.update(label=f"🔍 2단계: '{article_number}' 기반 원본 [별표] 처분표 추적 중...", state="running")
                docs_pass_2 = retriever.invoke(f"{article_number} {legal_keywords} 별표 행정처분 과태료 기준표")
                st.write("✔️ 2단계: 처분 기준표 데이터 확보 완료")

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
