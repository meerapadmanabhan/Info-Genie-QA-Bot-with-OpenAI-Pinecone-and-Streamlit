import streamlit as st
from langchain_helper import get_few_shot_db_chain

# Custom CSS to change the background color to pink
st.markdown(
    """
    <style>
    .css-18e3th9 {  /* This class name might change; use browser developer tools to inspect */
        background-color: #FFC0CB; /* Pink color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Myntra Online Store: Database Q&A 👕👗")

question = st.text_input("Question: ")

if question:
    chain = get_few_shot_db_chain()
    response = chain.run(question)

    st.header("Answer")
    st.write(response)
