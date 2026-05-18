import streamlit as st
import pandas as pd

st.set_page_config(page_title="Amazon AI Dashboard", layout="wide")

st.title("🛒 Amazon AI Dashboard")

df = pd.read_csv("amazon.csv")

# 검색창
keyword = st.text_input("상품 검색")

# 검색 결과
if keyword:
    filtered_df = df[df['product_name'].str.contains(keyword, case=False, na=False)]
else:
    filtered_df = df

st.subheader("검색 결과")
st.dataframe(filtered_df.head(20))
