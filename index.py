import signal

if not hasattr(signal, "_original_signal"):
    signal._original_signal = signal.signal
    def _safe_signal(sig, handler):
        import threading
        return handler if threading.current_thread() is not threading.main_thread() else signal._original_signal(sig, handler)
    
    signal.signal = _safe_signal


from datetime import datetime
from utils import web_search
from utils import getGuidanceDocs, processUploadedFiles, validateCTandSAP, save_uploaded_file_temp
from utils import getEvaluationScratchpad, getMatchedDocs, getMetadataQuery, getMetadataPrompt
from utils import evaluateResponse, convert_chat_history_to_pdf, getEvaluationQuery, getRecommendationQuery, getRecommendationScratchpad
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

import streamlit as st
import os

showWarningOnDirectExecution = False
style_subheader = "font-size: 24px; font-weight:bold; background-color:#F68830; color:white; padding-left:10px; border-radius: 5px;"

api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY is not set in environment variables or Streamlit secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

col1, col2 = st.columns([2, 6])
with col1:
    st.image("./Logo/ProtocolReviewLogo.png", width=150)

with col2:
    st.markdown("""<br/><h1 style="font-size: 3em; margin: 0; line-height:0.5;">PEARL</h1>
                <h3 style="font-size: 1.2em; margin: 0; line-height:0.5;">Protocol Evaluation and Recommendation with LLM</h3>""", unsafe_allow_html=True)


################################################### Session State ###################################################
defaults = {"selected_files": [], "local_files": [], "chat_history": [], "filenames": []}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


st.markdown(f"""<div style='{style_subheader}'>Upload Clinical Trial Protocol and Guidances</div><br/>""", unsafe_allow_html=True)

directoryFDA = "Guidances/FDA"
directoryICH = "Guidances/ICH"
directoryEMA = "Guidances/EMA"
directoryPMDA = "Guidances/PMDA"
directoryGF = "Guidances/GF"

guidancesFDA = getGuidanceDocs(directoryFDA)
guidancesICH = getGuidanceDocs(directoryICH)
guidancesEMA = getGuidanceDocs(directoryEMA)
guidancesPMDA = getGuidanceDocs(directoryPMDA)
guidancesGF = getGuidanceDocs(directoryGF)


col1, col2 = st.columns([5, 3])
################################################### select guidance source ###################################################
with col1:
    st.markdown(f"""<div style='font-size: 18px; font-weight:bold;'>Guidance Document Library</div><br/>""", unsafe_allow_html=True)
    source = st.radio("Select upload source:", ("FDA", "EMA", "PMDA", "ICH", "GF", "Local"), horizontal=True) ##### GATES
    if source == "FDA":
        file = st.selectbox("Select a PDF file:", guidancesFDA, index=None)
        if file and os.path.join(directoryFDA, file) not in st.session_state.selected_files:
            st.session_state.selected_files.append(os.path.join(directoryFDA, file))
    elif source == "ICH":
        file = st.selectbox("Select a PDF file:", guidancesICH, index=None)
        if file and os.path.join(directoryICH, file) not in st.session_state.selected_files:
            st.session_state.selected_files.append(os.path.join(directoryICH, file))
    elif source == "PMDA":
        file = st.selectbox("Select a PDF file:", guidancesPMDA, index=None)
        if file and os.path.join(directoryPMDA, file) not in st.session_state.selected_files:
            st.session_state.selected_files.append(os.path.join(directoryPMDA, file))
    elif source == "EMA":
        file = st.selectbox("Select a PDF file:", guidancesEMA, index=None)
        if file and os.path.join(directoryEMA, file) not in st.session_state.selected_files:
            st.session_state.selected_files.append(os.path.join(directoryEMA, file))
    elif source == "GF":
        file = st.selectbox("Select a PDF file:", guidancesGF, index=None)
        if file and os.path.join(directoryGF, file) not in st.session_state.selected_files:
            st.session_state.selected_files.append(os.path.join(directoryGF, file))
    elif source == "Local":
        reference_files = st.file_uploader("Clinical Trial Protocols Upload", type=["pdf"], accept_multiple_files=True, key=f"reference_files")
        if reference_files and reference_files not in st.session_state.local_files:
            for rf in reference_files:
                file_path = save_uploaded_file_temp(rf)
                st.session_state.local_files.append(file_path)
                st.session_state.filenames.append(rf.name)

################################################### Upload and Process CT and SAP ###################################################
with col2:
    st.markdown("<div style='font-size: 18px; font-weight:bold;'>Clinical Trial Protocol and SAP</div><br/>", unsafe_allow_html=True)
    package_files = st.file_uploader("", type=["pdf"], accept_multiple_files=True, key="package_files")
    msg, allCTChunks, table_docs = validateCTandSAP(package_files)

    if msg:
        st.write(msg)

################################################### Reset Selection ###################################################
if st.button("Reset Selection"):
    st.session_state.selected_files = []
    st.session_state.local_files = []
    st.session_state.filenames = []

    for file_path in st.session_state.local_files:
        try:
            os.remove(file_path)
        except OSError:
            pass


st.markdown("""<br/>""", unsafe_allow_html=True)

################################################### Display selected files ###################################################
markdown_text = ""
for file in st.session_state.selected_files:
    markdown_text += f"<div style='background-color: #f0f0f0; padding: 5px;'>{file.split('/')[-1]}</div><br/>"

for file in st.session_state.filenames:
    markdown_text += f"<div style='background-color: #f0f0f0; padding: 5px;'>{file.split('/')[-1]}</div><br/>"

st.markdown(f"""<div style='background-color: #f0f0f0; border-radius: 8px; padding: 5px;'>
            <div style='font-size: 18px; font-weight:bold;'>Guidance Document Selection</div>
            {markdown_text}</div><br/>""",unsafe_allow_html=True)



################################################### Process uploaded files ###################################################
allRefChunks = processUploadedFiles(st.session_state.selected_files, st.session_state.local_files)



################################################### Get Results ###################################################
# use_web = st.checkbox("Include Web Search (Live Internet Results)")
# st.session_state["use_web"] = use_web 

if st.button("Submit"):
    with st.spinner("Processing..."):
        llm = ChatOpenAI(model="gpt-4o", temperature=0, n = 1, streaming = True)
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory()

        if "chat" not in st.session_state:
            st.session_state.chat = ConversationChain(
                llm=llm,
                memory=st.session_state.memory
            )

        ################################################### Get Metadata ###################################################

        _, protocol_chunks = getMatchedDocs(allCTChunks, [], getMetadataQuery(), table_docs)

        prompt = getMetadataPrompt(protocol_chunks, getMetadataQuery())
        try:
            
            metadataResponse = llm.predict(prompt)
            dfMetadata = evaluateResponse(getMetadataQuery(), metadataResponse, protocol_chunks)
        except:
            dfMetadata = None
            metadataResponse = "Something went wrong! Could not generate a response."

        ################################################### Get Evaluation ###################################################
        guidance_chunks, protocol_chunks = getMatchedDocs(allCTChunks, allRefChunks, getEvaluationQuery(), table_docs)
        prompt = f"""You are an expert with 20 years of regulatory experience reviewing clinical trial protocols. Answer based only on the given context.

## Context:
**Protocol Excerpts**:
{'\n\n'.join(protocol_chunks)}

**FDA Guidance Excerpts**:
{'\n\n'.join(guidance_chunks)}

## Question:
{getEvaluationQuery()}

---

### 🔍 Scratchpad (Reason step-by-step):

{getEvaluationScratchpad()}

---

### 💬 Reference for missing:

missing 1: Page 12 of the SAP states that all below the limit of quantification (BLQ) values after a quantifiable concentration will be considered missing, but would this be a problem for PK parameter estimation if some of these values were truly BLQ?  

missing 2: The text on page 14 of the SAP suggests that a significant sequence effect will be ignored regardless of its magnitude. As a fundamental assumption of a crossover study is that the participants are in the same state at the start of each treatment period, some investigation may be warranted. We wonder whether it would be more appropriate to state something like “No carryover effects or sequence effects are anticipated given the washout period. However, if there is strong evidence of such effects then the cause of this will be investigated”.  

missing 3: The current FIH study will provide PK and tolerability data, as well as how much is retained in the nostrils and for how long. The ability to predict a protective dose will need to consider the results from the animal challenge studies.  

"""
        evaluationResponse = ""
        try:
            chat = st.session_state.chat
            evaluationResponse = chat.predict(input = prompt)
            dfEval = evaluateResponse(getEvaluationQuery(), evaluationResponse, guidance_chunks + protocol_chunks)
        except:
            dfEval = None
            evaluationResponse = "Something went wrong! Could not generate a response."

        ################################################### Get Recommendations ###################################################    
        prompt = f"""You are an expert with 20 years of regulatory experience reviewing clinical trial protocols. Answer based only on the given context.

## Response:
{evaluationResponse}

## Question:
{getRecommendationQuery()}

---

### 🔍 Scratchpad (Reason step-by-step):

{getRecommendationScratchpad()}

---

### 💬 Reference for missing:

improvement 1: We recommend incorporating the proposed data presentation format. In cases where a distinct trend is not apparent, consider obtaining a cefixime sample concurrent with serologic response sample collection (at the end of the dosage regimen) in forthcoming studies. This approach enables a correlation between systemic concentrations and serologic response, facilitating the modeling of anticipated placental concentrations derived from the maternal systemic concentration. (bp 10).

improvement 2: We also recommend following the FDA guideline on BA/BE studies (FDA 2014) that recommends that if the pre-dose value is > 5 percent of Cmax, the subject should be dropped from all PK evaluations. The subject data should be reported and the subject should be included in safety evaluations. 

improvement 3: We recommend providing further justification for the removal of BLQ values from the PK analyses. 

"""
        recommendationResponse = ""
        try:
            chat = st.session_state.chat
            recommendationResponse = chat.predict(input = prompt)
            dfRecomm = evaluateResponse(getRecommendationQuery(), recommendationResponse, [evaluationResponse])
        except:
            dfRecomm = None
            recommendationResponse = "Something went wrong! Could not generate a response."

        ################################################### Append to Chat History ###################################################
        st.session_state.chat_history.append([{"user": getMetadataQuery(), "assistant": metadataResponse}, 
                                              {"user": getEvaluationQuery(), "assistant": evaluationResponse}, 
                                              {"user": getRecommendationQuery(), "assistant": recommendationResponse}])


        # ################################################### Display results ###################################################
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Metadata", "SAP Evaluation", "SAP Recommendation", "PK/PD", "MIDD"])

        with tab1:
            st.header("Protocol and SAP Summary")
            st.write(metadataResponse)
            # AB: the results of metadata overview are different for different protocols. do we want to make it consistent across all protocols? 
            if not dfMetadata.empty:
                st.write("\nClinPharm:")
                st.markdown(f"""<table style="border-radius:5px; border: 1px solid #ddd;">
    <tr>
        <th width=20%>Metric</th>
        <th width=15%>Score (%)</th>
        <th width=65%>Reason</th>
    </tr>
    <tr>
        <td>{dfMetadata.iloc[0, 0]}</td>
        <td>{dfMetadata.iloc[0, 1]}</td>
        <td>{dfMetadata.iloc[0, 2]}</td>
    </tr>
    <tr>
        <td>{dfMetadata.iloc[1, 0]}</td>
        <td>{dfMetadata.iloc[1, 1]}</td>
        <td>{dfMetadata.iloc[1, 2]}</td>
    </tr>
    <tr>
        <td>{dfMetadata.iloc[2, 0]}</td>
        <td>{dfMetadata.iloc[2, 1]}</td>
        <td>{dfMetadata.iloc[2, 2]}</td>
    </tr>
</table>""", unsafe_allow_html=True)



        with tab2:
            st.header("SAP Evaluation")
            st.write(evaluationResponse)

            if not dfEval.empty:
                st.write("\nClinPharm:")
                st.markdown(f"""<table style="border-radius:5px; border: 1px solid #ddd;">
    <tr>
        <th width=20%>Metric</th>
        <th width=15%>Score (%)</th>
        <th width=65%>Reason</th>
    </tr>
    <tr>
        <td>{dfEval.iloc[0, 0]}</td>
        <td>{dfEval.iloc[0, 1]}</td>
        <td>{dfEval.iloc[0, 2]}</td>
    </tr>
    <tr>
        <td>{dfEval.iloc[1, 0]}</td>
        <td>{dfEval.iloc[1, 1]}</td>
        <td>{dfEval.iloc[1, 2]}</td>
    </tr>
    <tr>
        <td>{dfEval.iloc[2, 0]}</td>
        <td>{dfEval.iloc[2, 1]}</td>
        <td>{dfEval.iloc[2, 2]}</td>
    </tr>
</table>""", unsafe_allow_html=True)


        with tab3:
            st.header("SAP Recommendation")
            st.write(recommendationResponse)
            if not dfRecomm.empty:
                st.write("\nClinPharm:")
                st.markdown(f"""<table style="border-radius:5px; border: 1px solid #ddd;">
    <tr>
        <th width=20%>Metric</th>
        <th width=15%>Score (%)</th>
        <th width=65%>Reason</th>
    </tr>
    <tr>
        <td>{dfRecomm.iloc[0, 0]}</td>
        <td>{dfRecomm.iloc[0, 1]}</td>
        <td>{dfRecomm.iloc[0, 2]}</td>
    </tr>
    <tr>
        <td>{dfRecomm.iloc[1, 0]}</td>
        <td>{dfRecomm.iloc[1, 1]}</td>
        <td>{dfRecomm.iloc[1, 2]}</td>
    </tr>
    <tr>
        <td>{dfRecomm.iloc[2, 0]}</td>
        <td>{dfRecomm.iloc[2, 1]}</td>
        <td>{dfRecomm.iloc[2, 2]}</td>
    </tr>
</table>""", unsafe_allow_html=True)

        with tab4:
            st.image("./Logo/WIP.gif")

        with tab5: 
            st.image("./Logo/WIP.gif")



################################################### Download Chat History ###################################################
# Download Button for Chat History
# if st.session_state.chat_history:
#         current_date = datetime.now().date()
#         ## timestamp | protocols selected | Guidances | Scratchpad instructions | LLM used |Export to CSV?

#         st.markdown(f"""<br/><div style='{style_subheader}'>Download Chat History</div>""", unsafe_allow_html=True)
#         pdf_stream = convert_chat_history_to_pdf(st.session_state.chat_history)
#         if pdf_stream:
#             st.download_button("Download Chat History", pdf_stream, file_name=f"{current_date}_chat_history.pdf", mime="application/pdf")





        # ################################################### Get live results ###################################################
        # # if use_web:
        # #     try:
        # #         retrieved_docs += f"\n\n[Web Search Info]:\n{web_search(query)}"
        # #     except Exception as e:
        # #         st.warning(f"Web search failed: {e}")
        

################################################### Chat History ###################################################
# st.markdown(f"""<br/><div style='{style_subheader}'>Chat History</div>""", unsafe_allow_html=True)
# for chat in st.session_state.chat_history:
#     st.write(f"""**👤**: {chat['user']} \n\n**🤖**: {chat['assistant']}""")
#     st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True) 

# if st.session_state.chat_history:
#     if st.button("Clear Chat History"):
#         st.session_state.chat_history = []




