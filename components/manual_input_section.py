import streamlit as st
import pandas as pd


def manual_input():

    st.subheader("Manual Data Entry")

    date = st.date_input("Diagnosis Date")

    tests = st.number_input("Number Tested",0,100000)

    positives = st.number_input("Positive Cases",0,100000)

    if tests > 0:
        positivity = positives / tests
    else:
        positivity = 0

    df = pd.DataFrame({
        "Date_of_diagnosis":[date],
        "tested":[tests],
        "positives":[positives],
        "positivity_rate":[positivity]
    })

    return df