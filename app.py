import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------
# 페이지 설정
# ----------------------------------------
st.set_page_config(
    page_title="Y2K Seller Assistant",
    layout="wide"
)

# ----------------------------------------
# Y2K 스타일 CSS
# ----------------------------------------
st.markdown("""
<style>

/* 전체 배경 */
.stApp {
    background-color: #F6F4FF;
    color: #4B4B6A;
}

/* 사이드바 */
section[data-testid="stSidebar"] {
    background-color: #D9D6FF;
    border-right: 3px solid #B8B3FF;
}

/* 사이드바 글씨 */
section[data-testid="stSidebar"] * {
    color: #4B4B6A !important;
}

/* 제목 */
h1 {
    color: #6C63FF;
    font-size: 3.2rem;
}

/* 소제목 */
h2, h3 {
    color: #7A74D1;
}

/* 카드 */
div[data-testid="metric-container"] {
    background-color: white;
    border-radius: 25px;
    padding: 20px;
    border: 3px solid #C9C5FF;
    box-shadow: 4px 4px 12px rgba(0,0,0,0.08);
}

/* 입력창 */
.stTextInput input {
    background-color: white !important;
    border-radius: 18px !important;
    border: 2px solid #B8B3FF !important;
    color: #4B4B6A !important;
}

/* selectbox */
.stSelectbox div[data-baseweb="select"] {
    background-color: white !important;
    border-radius: 18px !important;
}

/* number input */
.stNumberInput input {
    background-color: white !important;
    color: #4B4B6A !important;
}

/* dataframe */
[data-testid="stDataFrame"] {
    border-radius: 20px;
    overflow: hidden;
}

/* 버튼 */
.stButton button {
    background-color: #AFA8FF !important;
    color: white !important;
    border-radius: 18px !important;
    border: none !important;
    font-weight: bold;
}

/* slider */
.stSlider {
    color: #AFA8FF !important;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# 제목
# ----------------------------------------
st.title("🎮 Y2K Seller Assistant")

st.markdown("""
감성적인 판매 전략 AI 도우미 ✨  
비슷한 상품 데이터를 분석하여 최적 할인 전략을 추천합니다.
""")

# ----------------------------------------
# 데이터 로드
# ----------------------------------------
df = pd.read_csv("amazon.csv")

# ----------------------------------------
# 데이터 전처리
# ----------------------------------------
df['discounted_price'] = (
    df['discounted_price']
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

numeric_cols = [
    'discounted_price',
    'discount_percentage',
    'rating',
    'rating_count'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=numeric_cols)

# 소비자 반응 점수
df['consumer_score'] = (
    df['rating'] *
    np.log1p(df['rating_count'])
)

# 텍스트 결합
df['combined_text'] = (
    df['product_name'].fillna('') + ' ' +
    df['about_product'].fillna('') + ' ' +
    df['category'].fillna('')
)

# ----------------------------------------
# TF-IDF
# ----------------------------------------
vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = vectorizer.fit_transform(
    df['combined_text']
)

# ----------------------------------------
# 사이드바
# ----------------------------------------
st.sidebar.header("🧸 상품 설정")

product_type = st.sidebar.radio(
    "상품 유형",
    ["신상품", "현재 판매 중 상품"]
)

# ========================================
# 신상품 모드
# ========================================
if product_type == "신상품":

    st.sidebar.subheader("✨ 신상품 정보")

    new_name = st.sidebar.text_input(
        "상품명",
        "Wireless Earbuds"
    )

    new_keywords = st.sidebar.text_input(
        "설명 키워드",
        "bluetooth noise cancelling"
    )

    category_list = sorted(
        df['category'].dropna().unique()
    )

    new_category = st.sidebar.selectbox(
        "카테고리",
        category_list
    )

    new_price = st.sidebar.number_input(
        "예상 판매 가격",
        min_value=0.0,
        value=5000.0
    )

    # 유사 상품 개수
    similar_count = st.sidebar.slider(
        "유사 상품 고려 개수",
        3,
        20,
        10
    )

    # 사용자 텍스트 생성
    user_text = (
        new_name + ' ' +
        new_keywords + ' ' +
        new_category
    )

    # 유사도 계산
    user_vector = vectorizer.transform(
        [user_text]
    )

    similarity_scores = cosine_similarity(
        user_vector,
        tfidf_matrix
    ).flatten()

    df['similarity'] = similarity_scores

    # 카테고리 필터
    similar_products = df[
        df['category'] == new_category
    ]

    # 가격 범위
    price_min = new_price * 0.7
    price_max = new_price * 1.3

    similar_products = similar_products[
        (similar_products['discounted_price'] >= price_min) &
        (similar_products['discounted_price'] <= price_max)
    ]

    # 유사도 정렬
    similar_products = similar_products.sort_values(
        by='similarity',
        ascending=False
    )

    top_products = similar_products.head(
        similar_count
    )

# ========================================
# 판매 중 상품 모드
# ========================================
else:

    st.sidebar.subheader("🔥 추천 검색 품목")

    recommended_items = [
        "smartphone",
        "earbuds",
        "keyboard",
        "mouse",
        "tablet",
        "speaker",
        "charger",
        "smartwatch"
    ]

    selected_item = st.sidebar.selectbox(
        "추천 품목",
        recommended_items
    )

    search_keyword = st.sidebar.text_input(
        "상품 검색",
        selected_item
    )

    search_result = df[
        df['product_name']
        .str.contains(
            search_keyword,
            case=False,
            na=False
        )
    ].head(15)

    product_options = []

    for idx, row in search_result.iterrows():

        option = (
            f"[{len(product_options)+1}] "
            f"{row['product_name'][:60]}"
        )

        product_options.append(option)

    selected_option = st.sidebar.selectbox(
        "분석 상품 선택",
        product_options
    )

    selected_idx = product_options.index(
        selected_option
    )

    selected_product = (
        search_result.iloc[selected_idx]
    )

    product_name = (
        selected_product['product_name']
    )

    product_category = (
        selected_product['category']
    )

    product_price = (
        selected_product['discounted_price']
    )

    top_products = df[
        df['category'] == product_category
    ]

# ----------------------------------------
# 시장 분석
# ----------------------------------------
market_discount = (
    top_products['discount_percentage']
    .mean()
)

market_rating = (
    top_products['rating']
    .mean()
)

market_review = (
    top_products['rating_count']
    .mean()
)

# ----------------------------------------
# 메인 분석
# ----------------------------------------
st.header("📊 판매 전략 분석")

c1, c2, c3 = st.columns(3)

c1.metric(
    "추천 할인율",
    f"{market_discount:.1f}%"
)

c2.metric(
    "시장 평균 평점",
    f"{market_rating:.2f}"
)

c3.metric(
    "시장 평균 리뷰 수",
    f"{market_review:.0f}"
)

# ----------------------------------------
# 시장 분석
# ----------------------------------------
st.subheader("🧠 시장 분석")

competition_level = len(top_products)

if competition_level >= 100:

    st.error("""
경쟁 강도가 매우 높은 시장이에요 ⚠️
할인 전략이 중요합니다.
""")

elif competition_level >= 40:

    st.warning("""
중간 수준 경쟁 시장입니다 ✨
리뷰 확보 전략이 중요합니다.
""")

else:

    st.success("""
경쟁 강도가 낮은 시장입니다 🌈
신규 진입 가능성이 있습니다.
""")

# ----------------------------------------
# 유사 상품 출력
# ----------------------------------------
st.header("🎀 참고 상품")

show_cols = [
    'product_name',
    'discounted_price',
    'discount_percentage',
    'rating',
    'rating_count'
]

st.dataframe(
    top_products[show_cols]
)

# ----------------------------------------
# 그래프
# ----------------------------------------
st.subheader("📉 할인율 비교")

chart_df = top_products[
    ['product_name',
     'discount_percentage']
]

chart_df = chart_df.set_index(
    'product_name'
)

st.bar_chart(chart_df)
