import streamlit as st


def show_table(df, rows):

    cols = ["ds","yhat"]

    # Prophet outputs include uncertainty intervals
    if "yhat_lower" in df.columns and "yhat_upper" in df.columns:
        cols += ["yhat_lower","yhat_upper"]

    st.dataframe(df[cols].tail(rows))


def display_results(results):

    st.header("Forecast Results")

    tab1,tab2,tab3 = st.tabs([
        "Daily Cases",
        "Weekly Cases",
        "Positivity Rate"
    ])


    with tab1:

        st.line_chart(results["daily"].set_index("ds")["yhat"])

        show_table(results["daily"],30)


    with tab2:

        st.line_chart(results["weekly"].set_index("ds")["yhat"])

        show_table(results["weekly"],12)


    with tab3:

        st.line_chart(results["positivity"].set_index("ds")["yhat"])

        show_table(results["positivity"],30)