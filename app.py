"""ממשק משתמש. הרצה: streamlit run app.py"""
import streamlit as st

from rag import ask, create_query_engine

st.set_page_config(page_title="עוזרת הניקיון", page_icon="🧽")
st.markdown("<style>.stApp {direction: rtl; text-align: right;}</style>", unsafe_allow_html=True)
st.title("🧽 עוזרת הניקיון")
st.write("שאלי שאלה בענייני ניקיון, למשל: איך מנקים ספת עור?")


@st.cache_resource
def get_engine():
    return create_query_engine()


question = st.text_input("השאלה שלך:")

if st.button("שאלי") and question:
    with st.spinner("מחפשת תשובה..."):
        answer, sources = ask(get_engine(), question)

    st.subheader("תשובה")
    st.write(answer)

    if sources:
        st.subheader("מקורות")
        for s in sources:
            with st.expander(f"{s['file']} (דמיון: {s['score']})"):
                st.write(s["text"])
