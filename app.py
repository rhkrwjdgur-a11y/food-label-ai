# --- 핵심 RAG 분석 로직 (시각적 진행률 표시 탑재) ---
@st.cache_resource(show_spinner=False)
def load_and_index_documents(_file_list, api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # 💡 이미 만들어진 뇌(DB)가 폴더에 있다면 1초 컷으로 로드!
    if os.path.exists(DB_PATH):
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    documents = []
    
    # ✨ 화면에 실시간 진행률 바 생성
    progress_bar = st.progress(0, text="🧠 AI 뇌(DB) 최초 생성 중... (데이터가 방대하여 5~10분 소요됩니다)")

    for i, file_path in enumerate(_file_list):
        # 현재 어떤 파일을 읽고 있는지 화면에 텍스트와 게이지로 표시
        progress_bar.progress((i + 1) / len(_file_list), text=f"[{i+1}/{len(_file_list)}] 📄 '{file_path}' 정독 중...")
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

    progress_bar.progress(1.0, text="✅ 파일 읽기 완료! 구글 AI로 데이터 변환(임베딩) 중... (1~2분 추가 소요)")

    if not documents:
        return None

    # 텍스트 분할 및 벡터 DB 생성
    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    ).split_documents(documents)

    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(DB_PATH)  # 벡터 DB 로컬 저장
    
    progress_bar.empty()  # 작업이 완전히 끝나면 진행률 바 숨기기
    return vectorstore
