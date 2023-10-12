# Document Based Question-Answering
This is a small project to learn document based QnA with LLMs using `langchain` and vector databases like `chromadb`. 

# Run
1. Create `.env` file with your `OPENAI_API_KEY`.
2. Install `requirements.txt`:
```bash
pip install -r requirements.txt
```
3. Run `main.py`:
```bash
chainlit run main.py
```
4. Open `http://localhost:8000` in your browser.
5. Upload PDF or text documents, and ask questions.