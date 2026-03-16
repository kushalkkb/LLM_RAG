py -3.11 -m venv venv311 - to create environment with python -3.11 version
venv311\Scripts\activate - to activate environment
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2  - offline model pull
deactivate - to deactivate
python rag_pdf_chat.py - to run the project

libraries required-
ollama pull llama3
pip install python-pptx
pip install -U langchain langchain-community langchain-text-splitters
pip install -U langchain-huggingface langchain-ollama
pip install -U faiss-cpu sentence-transformers pypdf2