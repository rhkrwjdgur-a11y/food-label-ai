# FAISS 저장/불러오기 (제미나이 아이디어) ✅
@st.cache_resource
def load_and_index_documents(_file_list, api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if os.path.exists(DB_PATH):
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    documents = []
    for file_path in _file_list:
        try:
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.lower().endswith(('.xls', '.xlsx')):
                loader = UnstructuredExcelLoader(file_path)
            elif file_path.lower().endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                continue
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"⚠️ {file_path} 로딩 실패: {e}")

    if not documents:
        return None

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(documents)

    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(DB_PATH)  # ✅ 저장
    return vectorstore


# 분석 실행 버튼 부분 - flash + 스트리밍 (내 제안) ✅
if st.button("분석 실행", type="primary"):
    if not pre_uploaded_files:
        st.warning("⚠️ 학습할 문서가 없습니다.")
    elif not user_question:
        st.warning("⚠️ 질문을 입력해주세요.")
    else:
        with st.status("📂 문서 준비 중...", expanded=False) as status:
            vector_db = load_and_index_documents(tuple(pre_uploaded_files), google_api_key)
            status.update(label="✅ 준비 완료", state="complete")

        if vector_db:
            st.markdown("### 📊 분석 결과")
            result_placeholder = st.empty()
            full_text = ""

            retriever = vector_db.as_retriever(search_kwargs={"k": 4})
            prompt = PromptTemplate.from_template(TEMPLATE)

            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",  # ✅ pro → flash
                temperature=0,
                streaming=True             # ✅ 스트리밍
            )

            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt | llm | StrOutputParser()
            )

            for chunk in rag_chain.stream(user_question):
                full_text += chunk
                result_placeholder.markdown(full_text + "▌")

            result_placeholder.markdown(full_text)
