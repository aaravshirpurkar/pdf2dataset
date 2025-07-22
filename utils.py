import streamlit as st

def get_headers():
    return {
        "Authorization": f"Bearer {st.secrets['LLM_API_KEY']}",
        "Content-Type": "application/json"
    }
