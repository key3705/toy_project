import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="Seller Assistant",
    layout="wide"
)

# =========================================================
# Y2K CSS
# =========================================================
st.markdown("""
<style>

/* 전체 */
.stApp {
    background-color: #F6F4FF;
    color: #4A4A68;
    font-family: 'Trebuchet MS', sans-serif;
}

/* 메인 */
.block-container {
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* 사이드바 */
section[data-testid="stSidebar"] {
    background-color: #DCD6FF;
    border-right: 4px solid #B7B0FF;
}

/* 사이드바 글씨 */
section[data-testid="stSidebar"] * {
    color: #4A4A68 !important;
}

/* 제목 */
h1 {
    color: #625CD6;
    font-size: 3.4rem;
    font-weight: 800;
}

/* 소제목 */
h2, h3 {
    color: #7268D8;
}

/* 카드 */
div[data-testid="metric-container"] {
    background-color: white;
    border-radius: 24px;
    padding: 18px;
    border: 3px solid #C7C2FF;
    box-shadow: 4px 4px 12px rgba(0,0,0,0.08);
}

/* 입력창 */
.stTextInput input {
    background-color: white !important;
    border-radius: 20px !important;
    border: 3px solid #B7B0FF !important;
    color: #4A4A68 !important;
    font-size: 18px !important;
}

/* number input */
.stNumberInput input {
    background-color: white !important;
    color: #4A4A68 !important;
}

/* selectbox */
.stSelectbox div[data-baseweb="select"] {
    background-color: white !important;
    border-radius: 18px !important;
    border: 3px solid #B7B0FF !important;
}

/* dropdown */
div[role="listbox"] * {
    background-color: white !important;
    color: #4A4A68 !important;
}

/* 버튼 */
.stButton button {
    background-color: #9F98FF !important;
    color: white !important;
    border-radius: 20px !important;
    border: none !important;
    font-size: 18px !important;
    font-weight: bold !important;
    padding: 12px 20px !important;
}

/* slider */
.stSlider {
    color: #9F98FF !important;
}

/* dataframe */
[data-testid="stDataFrame"] {
    border-radius: 20px;
    overflow: hidden;
}

/* expander */
.streamlit-expanderHeader {
    background-color: #ECE9FF;
    border-radius: 15px;
}

/* info card */
.info-card {
    background-color: white;
    padding: 25px;
    border-radius: 25px;
    border: 3px solid #C7C2FF;
    box-shadow: 4px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* analysis card */
.analysis-card {
    background-color: white;
    padding: 25px;
    border-radius: 25px;
    border: 3px solid #D8D3FF;
    box-shadow: 4px 4px 12px rgba(0,0,0,0.08);
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# 제목
# =========================================================
st.title("🎮 Seller Assistant")

st.markdown("""
상품 데이터를 기반으로  
판매 전략을 분석합니다.
""")

# =========================================================
# 데이터 로드
# =========================================================
df = pd.read_csv("amazon.csv")

# =========================================================
# 데이터 전처리
# =========================================================
df = df.copy()

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

    df[col] = pd.to_numeric(
        df[col],
        errors='coerce'
    )

df = df.dropna(subset=numeric_cols)

# =========================================================
# 소비자 반응 점수
# =========================================================
df['consumer_score'] = (
    df['rating'] *
    np.log1p(df['rating_count'])
)

# =========================================================
# 카테고리 정리
# =========================================================
category_mapping = {
    "Computers&Accessories": "💻 Computer",
    "Electronics": "🎧 Electronics",
    "Home&Kitchen": "🏠 Home",
    "OfficeProducts": "🖨 Office",
    "MusicalInstruments": "🎵 Music",
    "Health&PersonalCare": "💖 Health",
    "HomeImprovement": "🛠 Tools",
    "Toys&Games": "🧸 Toys"
}

df['main_category'] = (
    df['category']
    .astype(str)
    .apply(lambda x: x.split('|')[0])
)

df['display_category'] = (
    df['main_category']
    .map(category_mapping)
    .fillna("✨ Other")
)

# =========================================================
# 텍스트 결합
# =========================================================
df['combined_text'] = (
    df['product_name'].fillna('') + ' ' +
    df['about_product'].fillna('') + ' ' +
    df['display_category'].fillna('')
)

# =========================================================
# TF-IDF
# =========================================================
vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = vectorizer.fit_transform(
    df['combined_text']
)

# =========================================================
# 사이드바
# =========================================================
st.sidebar.header("🧸 상품 설정")

product_type = st.sidebar.radio(
    "상품 유형",
    ["신상품", "현재 판매 중 상품"]
)

# =========================================================
# 신상품
# =========================================================
if product_type == "신상품":

    st.sidebar.subheader("✨ 신상품 정보")

    product_name = st.sidebar.text_input(
        "상품명",
        "Wireless Headset"
    )

    product_keywords = st.sidebar.text_input(
        "설명 키워드",
        "bluetooth noise cancelling"
    )

    category_list = sorted(
        df['display_category']
        .dropna()
        .unique()
    )

    selected_category = st.sidebar.selectbox(
        "카테고리",
        category_list
    )

    product_price = st.sidebar.number_input(
        "예상 판매 가격",
        min_value=0.0,
        value=5000.0
    )

    similar_count = st.sidebar.slider(
        "유사 상품 고려 개수",
        3,
        20,
        10
    )

# =========================================================
# 판매 중 상품
# =========================================================
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

    selected_category = (
        selected_product['display_category']
    )

    product_price = (
        selected_product['discounted_price']
    )

    similar_count = st.sidebar.slider(
        "유사 상품 고려 개수",
        3,
        20,
        10
    )

# =========================================================
# 분석 버튼
# =========================================================
analyze_button = st.sidebar.button(
    "🎀 분석 시작"
)

# =========================================================
# 분석 실행
# =========================================================
if analyze_button:

    with st.spinner("🎮 시장 데이터를 분석 중이에요..."):

        # -------------------------------------------------
        # 신상품 분석
        # -------------------------------------------------
        if product_type == "신상품":

            user_text = (
                product_name + ' ' +
                product_keywords + ' ' +
                selected_category
            )

            user_vector = vectorizer.transform(
                [user_text]
            )

            similarity_scores = cosine_similarity(
                user_vector,
                tfidf_matrix
            ).flatten()

            df['similarity'] = similarity_scores

            top_products = df[
                df['display_category']
                == selected_category
            ]

            price_min = product_price * 0.7
            price_max = product_price * 1.3

            top_products = top_products[
                (top_products['discounted_price'] >= price_min) &
                (top_products['discounted_price'] <= price_max)
            ]

            top_products = top_products.sort_values(
                by='similarity',
                ascending=False
            )

            top_products = top_products.head(
                similar_count
            )

        # -------------------------------------------------
        # 판매 중 상품 분석
        # -------------------------------------------------
        else:

            top_products = df[
                df['display_category']
                == selected_category
            ]

            price_min = product_price * 0.7
            price_max = product_price * 1.3

            top_products = top_products[
                (top_products['discounted_price'] >= price_min) &
                (top_products['discounted_price'] <= price_max)
            ]

            top_products = top_products.head(
                similar_count
            )

        # -------------------------------------------------
        # 시장 분석
        # -------------------------------------------------
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

        competition_level = len(top_products)

    # =====================================================
    # 분석 정보 카드
    # =====================================================
    st.header("📦 분석 정보")

    st.markdown(f"""
    <div class="info-card">

    <h3>🎀 현재 분석 대상</h3>

    <b>상품 유형</b> : {product_type}<br><br>

    <b>카테고리</b> : {selected_category}<br><br>

    <b>상품명</b> : {product_name}<br><br>

    <b>참고 상품 수</b> : {similar_count}개

    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # 메트릭
    # =====================================================
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

    # =====================================================
    # 경쟁 분석
    # =====================================================
    st.subheader("⚔️ 시장 경쟁 분석")

    if competition_level >= 100:

        st.error("""
경쟁 강도가 높은 시장입니다.

가격 경쟁 및 할인 전략이 중요합니다.
""")

    elif competition_level >= 40:

        st.warning("""
중간 수준 경쟁 시장입니다.

리뷰 확보 전략이 중요합니다.
""")

    else:

        st.success("""
경쟁 강도가 낮은 시장입니다.

신규 판매 진입 가능성이 있습니다.
""")

    # =====================================================
    # 할인 전략
    # =====================================================
    st.subheader("💡 할인 전략 추천")

    recommended_min = market_discount - 5
    recommended_max = market_discount + 5

    st.markdown(f"""
    <div class="analysis-card">

    <h3>추천 할인 전략</h3>

    <h2>{recommended_min:.1f}% ~ {recommended_max:.1f}%</h2>

    유사 상품들은 평균적으로  
    약 <b>{market_discount:.1f}% 할인율</b>에서  
    높은 소비자 반응을 보였습니다.

    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # 참고 상품
    # =====================================================
    st.header("🎀 참고 상품 데이터")

    show_cols = [
        'product_name',
        'display_category',
        'discounted_price',
        'discount_percentage',
        'rating',
        'rating_count'
    ]

    st.dataframe(
        top_products[show_cols]
    )

    # =====================================================
    # 할인율 그래프
    # =====================================================
    st.subheader("📉 할인율 비교")

    chart_df = top_products[
        ['product_name',
         'discount_percentage']
    ]

    chart_df = chart_df.set_index(
        'product_name'
    )

    st.bar_chart(chart_df)

    # =====================================================
    # 리뷰 수 그래프
    # =====================================================
    st.subheader("⭐ 리뷰 수 비교")

    review_chart = top_products[
        ['product_name',
         'rating_count']
    ]

    review_chart = review_chart.set_index(
        'product_name'
    )

    st.bar_chart(review_chart)

else:

    st.markdown("""
    ## 🎀 분석 시작 버튼을 눌러주세요

    왼쪽 메뉴에서 상품 정보를 입력한 뒤  
    `🎀 분석 시작` 버튼을 누르면  
    분석 결과가 나타납니다.
    """)
