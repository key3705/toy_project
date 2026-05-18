import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------
# 페이지 설정
# -----------------------------------
st.set_page_config(
    page_title="Amazon Seller Discount Optimizer",
    layout="wide"
)

st.title("🛒 Amazon Seller Discount Optimizer")
st.markdown("AI 기반 판매 전략 및 최적 할인율 추천 시스템")

# -----------------------------------
# 데이터 로드
# -----------------------------------
df = pd.read_csv("amazon.csv")

# -----------------------------------
# 데이터 전처리
# -----------------------------------
df = df.copy()

# 문자열 정리
df['discounted_price'] = (
    df['discounted_price']
    .astype(str)
    .str.replace('₹', '')
    .str.replace(',', '')
)

df['actual_price'] = (
    df['actual_price']
    .astype(str)
    .str.replace('₹', '')
    .str.replace(',', '')
)

df['discount_percentage'] = (
    df['discount_percentage']
    .astype(str)
    .str.replace('%', '')
)

df['rating_count'] = (
    df['rating_count']
    .astype(str)
    .str.replace(',', '')
)

# 숫자 변환
numeric_cols = [
    'discounted_price',
    'actual_price',
    'discount_percentage',
    'rating',
    'rating_count'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 결측 제거
df = df.dropna(subset=numeric_cols)

# -----------------------------------
# 소비자 반응 점수
# -----------------------------------
df['consumer_score'] = (
    df['rating'] * np.log1p(df['rating_count'])
)

# -----------------------------------
# 텍스트 결합
# -----------------------------------
df['combined_text'] = (
    df['product_name'].fillna('') + ' ' +
    df['about_product'].fillna('') + ' ' +
    df['category'].fillna('')
)

# -----------------------------------
# TF-IDF
# -----------------------------------
vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

# -----------------------------------
# 사이드바
# -----------------------------------
st.sidebar.header("📌 상품 정보 입력")

product_type = st.sidebar.radio(
    "상품 유형 선택",
    ["신상품", "현재 판매 중 상품"]
)

user_name = st.sidebar.text_input(
    "상품 이름",
    "Wireless Earbuds"
)

user_keywords = st.sidebar.text_input(
    "상품 설명 키워드",
    "bluetooth noise cancelling bass"
)

category_list = sorted(
    df['category'].dropna().unique()
)

user_category = st.sidebar.selectbox(
    "카테고리",
    category_list
)

user_price = st.sidebar.number_input(
    "판매 가격",
    min_value=0.0,
    value=5000.0
)

# -----------------------------------
# 판매 중 상품 추가 정보
# -----------------------------------
if product_type == "현재 판매 중 상품":

    current_discount = st.sidebar.slider(
        "현재 할인율 (%)",
        0,
        90,
        10
    )

    current_rating = st.sidebar.slider(
        "현재 평점",
        1.0,
        5.0,
        4.0
    )

    current_review_count = st.sidebar.number_input(
        "현재 리뷰 수",
        min_value=0,
        value=100
    )

# -----------------------------------
# 사용자 텍스트 생성
# -----------------------------------
user_text = (
    user_name + ' ' +
    user_keywords + ' ' +
    user_category
)

# -----------------------------------
# 유사도 계산
# -----------------------------------
user_vector = vectorizer.transform([user_text])

similarity_scores = cosine_similarity(
    user_vector,
    tfidf_matrix
)

similarity_scores = similarity_scores.flatten()

df['similarity'] = similarity_scores

# -----------------------------------
# 카테고리 필터
# -----------------------------------
filtered_df = df[
    df['category'] == user_category
]

# 가격대 필터
price_min = user_price * 0.7
price_max = user_price * 1.3

filtered_df = filtered_df[
    (filtered_df['discounted_price'] >= price_min) &
    (filtered_df['discounted_price'] <= price_max)
]

# 유사도 순 정렬
filtered_df = filtered_df.sort_values(
    by='similarity',
    ascending=False
)

top_products = filtered_df.head(10)

# -----------------------------------
# 추천 할인율 계산
# -----------------------------------
optimal_discount = top_products[
    'discount_percentage'
].mean()

average_rating = top_products[
    'rating'
].mean()

average_review_count = top_products[
    'rating_count'
].mean()

average_consumer_score = top_products[
    'consumer_score'
].mean()

# -----------------------------------
# 메인 결과
# -----------------------------------
st.header("📊 AI 판매 전략 분석")

col1, col2, col3 = st.columns(3)

col1.metric(
    "추천 할인율",
    f"{optimal_discount:.1f}%"
)

col2.metric(
    "시장 평균 평점",
    f"{average_rating:.2f}"
)

col3.metric(
    "시장 평균 리뷰 수",
    f"{average_review_count:.0f}"
)

# -----------------------------------
# 판매 중 상품 분석
# -----------------------------------
if product_type == "현재 판매 중 상품":

    st.subheader("📈 현재 판매 전략 평가")

    user_consumer_score = (
        current_rating *
        np.log1p(current_review_count)
    )

    market_score = average_consumer_score

    diff = user_consumer_score - market_score

    if diff > 5:
        strategy_result = "시장 평균보다 매우 좋은 반응"
        strategy_color = "🟢"

    elif diff > 0:
        strategy_result = "시장 평균보다 약간 좋은 반응"
        strategy_color = "🟡"

    else:
        strategy_result = "시장 평균 대비 개선 필요"
        strategy_color = "🔴"

    st.markdown(f"""
    ### {strategy_color} 현재 전략 분석

    - 현재 할인율: **{current_discount}%**
    - 현재 평점: **{current_rating}**
    - 현재 리뷰 수: **{current_review_count}**

    현재 상품의 소비자 반응은  
    **{strategy_result}** 상태입니다.
    """)

    # 할인 전략 추천
    st.subheader("💡 할인 전략 추천")

    if current_discount < optimal_discount:

        st.info(f"""
        현재 할인율이 시장 추천 할인율보다 낮습니다.

        추천:
        **{current_discount}% → {optimal_discount:.1f}%**
        수준으로 할인율 조정을 고려해볼 수 있습니다.
        """)

    elif current_discount > optimal_discount:

        st.warning(f"""
        현재 할인율이 시장 평균보다 높습니다.

        할인율을 일부 줄여도
        소비자 반응 유지 가능성이 있습니다.
        """)

    else:

        st.success("""
        현재 할인 전략이 시장 평균과 유사합니다.
        """)

# -----------------------------------
# 신상품 분석
# -----------------------------------
else:

    st.subheader("🧠 신상품 할인 전략 추천")

    st.write(f"""
    입력한 상품과 유사한 상품들을 분석한 결과,

    현재 카테고리와 가격대에서는  
    평균적으로 약 **{optimal_discount:.1f}% 할인율**에서
    높은 소비자 반응이 나타났습니다.

    추천 할인 전략:
    **약 {optimal_discount:.1f}% 수준**
    """)

# -----------------------------------
# 유사 상품
# -----------------------------------
st.subheader("🔍 유사 상품 TOP 10")

show_cols = [
    'product_name',
    'discounted_price',
    'discount_percentage',
    'rating',
    'rating_count',
    'similarity'
]

st.dataframe(
    top_products[show_cols]
)

# -----------------------------------
# 할인율 그래프
# -----------------------------------
st.subheader("📉 유사 상품 할인율 비교")

chart_df = top_products[
    ['product_name', 'discount_percentage']
].copy()

chart_df = chart_df.set_index('product_name')

st.bar_chart(chart_df)

# -----------------------------------
# 데이터 확인
# -----------------------------------
with st.expander("전체 데이터 보기"):
    st.dataframe(df.head(100))
