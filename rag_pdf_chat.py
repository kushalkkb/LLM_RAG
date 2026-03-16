from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


# -------------------------------
# 1. Load PDF
# -------------------------------
def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    return text


# -------------------------------
# 2. Split text into chunks
# -------------------------------
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)


# -------------------------------
# 3. Create Vector Store
# -------------------------------
def create_vector_store(chunks):

    # Load embedding model locally (OFFLINE)
    embeddings = HuggingFaceEmbeddings(
        model_name="./all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_texts(chunks, embeddings)
    return vector_db


# -------------------------------
# 4. Answer Question (Manual RAG)
# -------------------------------
def answer_question(vector_db, query):

    docs = vector_db.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    llm = OllamaLLM(model="llama3")

    prompt = f"""
You are an AI assistant answering questions based ONLY on the given document context.

Context:
{context}

Question:
{query}

Answer clearly and accurately:
"""

    return llm.invoke(prompt)


# -------------------------------
# 5. Main Chat Loop
# -------------------------------
def chat_with_document(pdf_path):

    print("📄 Loading document...")
    text = load_pdf(pdf_path)

    print("✂️ Splitting text...")
    chunks = split_text(text)

    print("🧠 Creating vector database...")
    vector_db = create_vector_store(chunks)

    print("\n✅ Document ready! Ask questions (type 'exit' to stop)\n")

    while True:

        query = input("You: ")

        if query.lower() == "exit":
            print("👋 Exiting chat...")
            break

        response = answer_question(vector_db, query)

        print("\nAI:", response, "\n")


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":

    pdf_file_path = r"C:\pc\PROJECT_DRIVE\PROJECTS\LLM_RAG\SCANSTORM.pdf"

    chat_with_document(pdf_file_path)