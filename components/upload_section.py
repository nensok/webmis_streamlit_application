import streamlit as st
import pandas as pd

def upload_data():

    uploaded_file = st.file_uploader(
        "Upload Surveillance Dataset",
        type=["csv","xlsx"]
    )

    if uploaded_file:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        else:
            df = pd.read_excel(uploaded_file)

        return df

    return None