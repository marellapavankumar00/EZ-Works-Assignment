
# File: app.py

import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
import random

# Load summarizer and QA pipeline
summarizer = pipeline("summarization")
qa_pipeline = pipeline("question-answering")

st.set_page_config(page_title="Smart Research Assistant", layout="wide")
st.title("🧑‍💻 Smart Assistant for Research Summarization")

# Function to extract text from PDF
def extract_text_from_pdf(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to generate summary
def generate_summary(text):
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summary = ""
    for chunk in chunks:
        try:
            out = summarizer(chunk, max_length=50, min_length=25, do_sample=False)
            summary += out[0]['summary_text'] + " "
        except:
            continue
    return summary.strip()

# Function to generate 3 logic-based questions
def generate_questions(text):
    sentences = list(set(text.split(".")))
    random.shuffle(sentences)
    questions = []
    for i in range(3):
        q = f"Based on this document, what is the meaning of: '{sentences[i].strip()[:80]}...?"
        questions.append((q, sentences[i].strip()))
    return questions

# Upload section
uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        doc_text = extract_text_from_pdf(uploaded_file)
    else:
        doc_text = uploaded_file.read().decode("utf-8")

    st.subheader("📄 Document Summary")
    summary = generate_summary(doc_text)
    st.info(summary[:150])

    mode = st.radio("Choose Interaction Mode:", ["Ask Anything", "Challenge Me"])

    if mode == "Ask Anything":
        user_question = st.text_input("Ask a question based on the document:")
        if user_question:
            answer = qa_pipeline(question=user_question, context=doc_text)
            st.success(f"**Answer:** {answer['answer']}")
            st.caption("Justification: Answer derived from the document using transformer QA model.")

    elif mode == "Challenge Me":
        st.write("Here are your 3 logic-based questions:")
        questions = generate_questions(doc_text)
        for idx, (q, ans) in enumerate(questions):
            user_ans = st.text_input(f"Q{idx+1}: {q}", key=idx)
            if user_ans:
                if user_ans.lower() in ans.lower():
                    st.success("Correct! ✅")
                else:
                    st.error(f"Not quite. ❌\nReference Answer: {ans}")
                    st.caption("Justification: This line appears directly in the document.")
