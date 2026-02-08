import streamlit as st
import requests

st.set_page_config(page_title="Grounded LLM Demo", layout="centered")

st.title("Grounded LLM System")
st.caption("Answers only when supported by source documents")

question = st.text_input(
    "Ask a medical question",
    placeholder="e.g. What is hypertension?"
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Checking documents and grounding answer..."):
            res = requests.post(
                "http://127.0.0.1:8000/ask",
                json={"question": question}
            ).json()

        if res["grounded"]:
            st.success("Grounded Answer")
            st.write(res["answer"])
        else:
            st.error("Safe Refusal")
            st.write(res["answer"])

        st.divider()
        st.caption(f"Grounded: {res['grounded']}")
