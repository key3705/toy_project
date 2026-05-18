import streamlit as st
import pandas as pd

st.title("Amazon AI Dashboard")

df = pd.read_csv("amazon.csv")

st.subheader("데이터 미리보기")
st.write(df.head())