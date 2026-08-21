import streamlit as st
from utils import extract_pdf, vector_creation
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Tanmay placement RAG")
st.title("Tanmay resume analyzer AI")

resume_file = st.file_uploader(
    "upload Resume (PDF)",
    type=["pdf"]
)

jd_text = st.text_area(
    "Paste the job description here "
)

if st.button("Analyze"):

    if resume_file:

        # Extracting the resume content
        resume_text = extract_pdf(resume_file)

        if not resume_text.strip():
            st.error("Could not extract text from the uploaded PDF.")
            st.stop()

        #combine resume and jd
        vector_store = vector_creation(resume_text)

        retriever = vector_store.as_retriever(
            search_kwargs={"k": 5}
        )

        #load LLM models and integrate
        llm = ChatOllama(
            model="gemma3:4b",
            temperature=0
        )

        #prompt template design
        prompt = ChatPromptTemplate.from_template(
            """
            You are an AI resume analyzer and placement coach.

            Analyze the candidate's resume against the provided job description.

            Your analysis must be:
            - Direct and honest
            - Technically accurate
            - Evidence-based
            - Constructive
            - Focused on improving the candidate's chances of getting shortlisted

            Do not invent or assume any skills, projects, qualifications,
            experience, or achievements that are not present in the resume.

            Resume Context:
            {context}

            Job Description:
            {Question}

            Provide the analysis in the following structure:

            1. Overall Match
            - Give an estimated match percentage.
            - Briefly explain the overall suitability of the candidate.

            2. Matching Skills
            - List the technical skills from the resume that directly match
              the requirements of the job description.
            - Mention relevant evidence from the resume where possible.

            3. Missing Skills
            - Identify important skills or technologies required by the job
              description that are missing from the resume.

            4. Relevant Experience and Projects
            - Identify projects, internships, work experience, or academic
              work that are relevant to the position.
            - Explain why each is relevant.

            5. Resume Weaknesses
            - Identify technical, structural, or presentation weaknesses.
            - Prioritize weaknesses that could reduce the candidate's
              chances of being shortlisted.

            6. Improvement Suggestions
            - Give specific and actionable suggestions.
            - Recommend skills, projects, certifications, or resume changes
              only when they are relevant to the job description.

            7. ATS Analysis
            - Identify important keywords from the job description that are
              missing or weakly represented in the resume.
            - Suggest where relevant keywords could naturally be incorporated.

            8. Final Recommendation
            - Classify the candidate as:
              Strong Match, Moderate Match, or Weak Match.
            - Explain the main reasons for the recommendation.

            Do not blindly praise the candidate.
            Do not criticize the candidate personally.
            Evaluate the resume professionally and objectively.
            """
        )

        chain = (
            {
                "context": retriever,
                "Question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(jd_text)

        st.subheader("Analysis result")
        st.write(response)

    else:
        st.warning("Please upload a resume PDF.")