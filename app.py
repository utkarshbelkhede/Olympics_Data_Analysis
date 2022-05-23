import streamlit as st
from analysis_page import analysis
from model_page import model_compare
from explore_page import explore
from predict_page import show_predict_page

st.sidebar.write("""
    ### Explore, Analysis, Compare Or Predict
    """)
page = st.sidebar.selectbox("", ("Explore","Olympics Data Analysis","Compare Models","Predict"))

if page == "Explore":
    explore()
elif page == "Olympics Data Analysis":
    analysis()
elif page == "Compare Models":
    model_compare()
elif page == "Predict":
    show_predict_page()