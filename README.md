# PEARL 2.0

PEARL (Protocol Evaluation and Review with LLMs) is a Streamlit-based app that uses retrieval-augmented generation (RAG) to assist with clinical protocol assessment.

## 🚀 Getting Started

Follow these steps to set up and run the app locally.

### 1. Clone the repository

```bash
git clone https://github.com/AmrutaIdagunji/PEARL.git
cd PEARL2.0
````

Or download the ZIP file and extract it.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

Set your API key as an environment variable or add it to `streamlit secrets`:

```bash
export OPENAI_API_KEY="your-api-key-here"
```


### 4. Run the app

```bash
streamlit run index.py
```

---

## 🔐 Note

Do **not** hardcode API keys in the source files. Use environment variables or `st.secrets`.

---

## 🧭 How to Use

Once the app launches in your browser:

### 1. **Select Guidance Documents**

Upload or select reference documents (e.g., FDA or ICH guidelines) for protocol evaluation. 

### 2. **Upload Clinical Protocols**

Upload one clinical trial protocols (PDFs only). These will be evaluated against the selected guidance documents.

### 3. **Submit for Evaluation**

Click the **Submit** button to start the evaluation. The LLM will generate a structured assessment based on RAG.

### 4. **Review Output**

The results will be displayed.

---

## 🧠 Built With

* [Streamlit](https://streamlit.io/)
* [LangChain](https://www.langchain.com/)
* [OpenAI API](https://platform.openai.com/)

---

## 📬 Contact

For questions or contributions, please reach out to **Amruta Bhat** or open an issue on the repository.

