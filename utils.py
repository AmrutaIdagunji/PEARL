def load_pdf(path):
    import tempfile
    from langchain_community.document_loaders import PyMuPDFLoader

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(path.read())
        tmp_path = tmp_file.name

    loader = PyMuPDFLoader(tmp_path, extract_images = True, images_inner_format = "text", extract_tables = "markdown")
    documents = loader.load()
    return documents  


def save_uploaded_file_temp(uploaded_file):
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name 

def chunk_documents(documents, source, chunk_size=200, chunk_overlap=0):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    ### https://python.langchain.com/docs/how_to/recursive_text_splitter/
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                              chunk_overlap=chunk_overlap,
                                              separators=["\n\n", "\n", " ", ".", ",",], 
                                              length_function=len)
    
    chunks = splitter.split_documents(documents)
    for doc in chunks:
        doc.metadata["doc_type"] = source 
    return chunks


def web_search(query, num_results=3):
    import serpapi
    params = {
        "engine": "google",
        "q": query,
        "api_key": "API",
        "num": num_results
    }
    search = serpapi.search(params)
    results = search.get_dict()
    snippets = []
    for result in results.get("organic_results", []):
        if "snippet" in result:
            snippets.append(result["snippet"])
    return " ".join(snippets)


def convert_chat_history_to_pdf(chat_history):
    from xhtml2pdf import pisa
    from markdown import markdown
    from io import BytesIO

    md_parts = []
    for i, chat in enumerate(chat_history, 1):
        md_parts.append(f"### Metadata {i}\n{chat[0]['user'].strip()}")
        md_parts.append(f"### Response {i}\n{chat[0]['assistant'].strip()}\n")

        md_parts.append(f"### Evaluation {i}\n{chat[1]['user'].strip()}")
        md_parts.append(f"### Response {i}\n{chat[1]['assistant'].strip()}\n")

        md_parts.append(f"### Recommendation {i}\n{chat[2]['user'].strip()}")
        md_parts.append(f"### Response {i}\n{chat[2]['assistant'].strip()}\n")


    markdown_text = "\n\n".join(md_parts)
    html_body = markdown(markdown_text, extensions=["extra", "sane_lists", "toc"])
    html = f"""
    <html>
    <head>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            padding: 5px;
            font-size: 12pt;
        }}
        h1, h2, h3, p, ul, ol, li {{
            padding: 5px;
        }}
        strong {{
            font-weight: bold;
        }}
        em {{
            font-style: italic;
        }}
    </style>
    </head>
    <body>{html_body}</body>
    </html>
    """

    pdf_buffer = BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=pdf_buffer)
    pdf_buffer.seek(0)
    
    return pdf_buffer if not pisa_status.err else None


def getGuidanceDocs(directory):
    import os
    guidances = [f for f in os.listdir(directory) if f.endswith('.pdf') and os.path.isfile(os.path.join(directory, f))]
    return guidances


def clean_whitespaces(text):
    import re
    return re.sub(r'\n+', ' ', text).strip()


def processUploadedFiles(selected_files, selected_local_files, chunk_size=500, chunk_overlap=0, source = "guidance"):
    from langchain_community.document_loaders import PyMuPDFLoader
    allRefChunks = []
    if selected_files or selected_local_files:
        for file in selected_files:
            loader = PyMuPDFLoader(file)
            documents = loader.load()
            for doc in documents:
                doc.page_content = clean_whitespaces(doc.page_content)

            refChunks = chunk_documents(documents, source=source, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            allRefChunks.extend(refChunks)

        for file in selected_local_files:
            loader = PyMuPDFLoader(file)
            documents = loader.load()
            for doc in documents:
                doc.page_content = clean_whitespaces(doc.page_content)

            refChunks = chunk_documents(documents, source=source, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            allRefChunks.extend(refChunks)

    return allRefChunks

def validateCTandSAP(package_files):
    allCTChunks = []
    table_docs = []
    msg = None
    try:
        if package_files:
            for file in package_files:
                protocol_docs = load_pdf(file)
                table = extract_tables(protocol_docs)
                ctChunks = chunk_documents(protocol_docs, source="protocol", chunk_size=500, chunk_overlap=0)
                allCTChunks.extend(ctChunks)
                table_docs.extend(table)

            msg = "Clinical Trial Protocol(s) processed successfully."
    except Exception as e:
        msg = "Error occured while processing the document(s)."
  
    return msg, allCTChunks, table_docs


# extract markdown table from loaded file
def extract_markdown_tables(text):
    import re
    pattern = r"(\|.+\|\n(\|[-:\s|]+\|\n)((\|.*\|\n?)+))"
    matches = re.findall(pattern, text)
    return [match[0] for match in matches]  # Each match[0] is full table text


# convert markdown table to dataframe
def markdown_to_dataframe(markdown: str):
    import pandas as pd
    from io import StringIO
    try:
        return pd.read_csv(StringIO(markdown), sep="|", engine='python').dropna(axis=1, how='all').dropna(axis=0, how='all')
    except Exception:
        return None


def extract_tables(file):
    from langchain_core.documents import Document
    markdown_tables = []
    for doc in file:
        for table in extract_markdown_tables(doc.page_content):
            markdown_tables.append(table)

    table_docs = [Document(page_content=table, metadata={"type": "table"}) for table in markdown_tables]
    return table_docs


def getEvaluationScratchpad():
    return """For each key statistical analysis section in the clinical trial protocol:
- Identify any aspects that align with ICH E9 principles.
- Note any gaps, unclear content, or concerns under Comment.
- Keep all observations grouped under each key section heading only. Do not create separate headers for comments.
"""

def getRecommendationScratchpad():
    return """For each gaps, unclear content, or concerns under 'Comment' in the key statistical analysis section of the clinical trial protocol:
- Provide an improvement, prefixed by its criticality (e.g., [Medium]).
- Keep all observations grouped under each key section heading. Do not create separate headers for recommendations.
"""

def  docs_to_dicts(docs):
    return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]


def search_vectorstore(store, query, k, fetch_k=None):
    return store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)


def getMatchedDocs(allCTChunks, allRefChunks, query, table_docs):

    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    if len(allRefChunks) != 0:
        # get the guidances
        vectorstore = FAISS.from_documents(allRefChunks, embedding_model)
        guidance_chunks = search_vectorstore(vectorstore, query, 50, len(allRefChunks))
        guidance_hits = [doc.page_content for doc in guidance_chunks]
    else:
        guidance_hits = []

    table_vectorstore = FAISS.from_documents(table_docs, embedding_model)
    text_vectorstore = FAISS.from_documents(allCTChunks, embedding_model)
    query_text = query + "\n\n".join(guidance_hits)

    protocol_table_chunks = table_vectorstore.similarity_search_with_score(query_text, k=10)
    protocol_text_chunks = text_vectorstore.similarity_search_with_score(query_text, k=50)

    combined = protocol_table_chunks + protocol_text_chunks
    combined_sorted = sorted(combined, key=lambda x: x[1]) 
    
    top_hits = combined_sorted[:50]
    protocol_hits = [doc.page_content for doc, _ in top_hits]
    return guidance_hits, protocol_hits


def evaluateResponse(query, response, retrieved_docs):
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    import pandas as pd

    ############ AB: check how does 
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model="gpt-3.5-turbo", include_reason=True, async_mode=False)
    faithfulness_metric = FaithfulnessMetric(threshold=0.7, model="gpt-4o", include_reason=True, async_mode=True)
    
    correctness_metric = GEval(
        name="Clinical Pharmacology",
        # https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation
        evaluation_steps=["Check whether the facts in 'actual output' contradicts clinical pharmacology principles",
                          "You should penalize omission of detail",
                          "Check whether the facts in 'actual output' are relevant to drug development"],   
                          ########## AB: what can be done
                          ########## AB: other drugs treat the same therapeutic class 
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
        threshold=0.7, model="gpt-4o", strict_mode=False, async_mode=False, verbose_mode=False
    )
    test_case = LLMTestCase(input=query, actual_output=response, retrieval_context= retrieved_docs)

    # Evaluate metrics
    answer_relevancy_metric.measure(test_case)
    faithfulness_metric.measure(test_case)
    correctness_metric.measure(test_case)
    
    #### evaluation speedometer
    data = {"Metric": ["Answer Relevancy", "Faithfulness", "ClinPharm Metric"],
            "Score (%)": [answer_relevancy_metric.score * 100, faithfulness_metric.score * 100, correctness_metric.score * 100], 
            "Reason": [answer_relevancy_metric.reason, faithfulness_metric.reason, correctness_metric.reason]}

    df = pd.DataFrame(data)
    df["Score (%)"] = df["Score (%)"].round(2)
    return df


def getMetadataQuery():
    return """Analyze and summarize the attached clinical trial protocol and statistical analysis plan."""


"""Analyze and summarize the attached clinical trial protocol and statistical analysis plan. Extract the following information:
Step 1: Clinical Trial Protocol Summary
- Study Title
- Study Design (type, population, randomization ratio, sample size etc.)
- Primary Objective
- Secondary Objectives (if any)
- Exploratory Objective (if any)
- Treatments (drug, dose, route, schedule)
- Endpoints (Primary / Secondary)
- Key Eligibility Criteria
- Study Phase and NCT Number (if available)

Step 2: Statistical Analysis Plan Summary
- Statistical Methods for Primary Endpoint
- Handling of Missing Data
- Interim Analysis / Sensitivity Analysis Details
- Subgroup Analyses
- Safety Analysis Methods

"""


def getMetadataPrompt(protocol_chunks, query):
    return f"""You are an expert at the United States Food and Drug Administration (FDA) with 20 years of regulatory experience reviewing clinical trial protocols.

Extract information as accurately as possible based on the context provided and structure your response into clearly labeled sections

User Query:
{query}

Context: 
{protocol_chunks}

Respond concisely but completely using bullet points and short paragraphs.
"""


def getEvaluationQuery():
    return """Based on the guidelines from the E9 Statistical Principles for Clinical Trials, evaluate the statistical analysis plan for the attached S2_Protocols&SAP"""


def getRecommendationQuery():
    return """Based on the attached evaluation of the document S2_Protocols&SAP, generate recommendations to improve the statistical analysis plan (SAP) in alignment with ICH E9: Statistical Principles for Clinical Trials."""

