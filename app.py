import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------
# 페이지 설정
# ---------------------------------------
st.set_page_config(
    page_title="AI Seller Strategy Platform",
    layout="wide"
)

# ---------------------------------------
# 제목
# ---------------------------------------
st.title("🛒 AI Seller Strategy Platform")
st.markdown(
    """
판매자를 위한 AI 기반 할인 전략 및 시장 분석 시스템  
비슷한 상품들의 소비자 반응 데이터를 분석하여
최적 할인 전략을 추천합니다.
"""
)

# ---------------------------------------
# 데이터 로드
# ---------------------------------------
df = pd.read_csv("amazon.csv")

# ---------------------------------------
# 데이터 전처리
# ---------------------------------------
df = df.copy()

# 문자열 처리
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
    df[col] = pd.to_numeric(
        df[col],
        errors='coerce'
    )

# 결측 제거
df = df.dropna(subset=numeric_cols)

# 소비자 반응 점수
df['consumer_score'] = (
    df['rating'] *
    np.log1p(df['rating_count'])
)

# ---------------------------------------
# 사이드바
# ---------------------------------------
st.sidebar.header("📌 상품 정보 입력")

# 상품 유형
product_type = st.sidebar.radio(
    "상품 유형 선택",
    ["신상품", "현재 판매 중 상품"]
)

# ---------------------------------------
# 추천 검색 품목
# ---------------------------------------
st.sidebar.subheader("🔥 추천 검색 품목")

recommended_items = [
    "smartphone",
    "earbuds",
    "keyboard",
    "mouse",
    "tablet",
    "speaker",
    "charger",
    "smartwatch",
    "headphones",
    "power bank"
]

selected_recommendation = st.sidebar.selectbox(
    "추천 품목 선택",
    recommended_items
)

# ---------------------------------------
# 검색
# ---------------------------------------
search_keyword = st.sidebar.text_input(
    "상품 검색",
    selected_recommendation
)

# ---------------------------------------
# 상품 검색 결과
# ---------------------------------------
search_result = df[
    df['product_name']
    .str.contains(
        search_keyword,
        case=False,
        na=False
    )
]

search_result = search_result.head(15)

# ---------------------------------------
# 검색 결과 표시
# ---------------------------------------
st.sidebar.subheader("🔍 검색 결과")

if len(search_result) == 0:

    st.sidebar.warning(
        "검색 결과가 없습니다."
    )

    st.stop()

# 상품 목록 생성
product_options = []

for idx, row in search_result.iterrows():

    option = (
        f"[{len(product_options)+1}] "
        f"{row['product_name'][:65]}"
    )

    product_options.append(option)

# 상품 선택
selected_option = st.sidebar.selectbox(
    "분석할 상품 선택",
    product_options
)

selected_idx = product_options.index(
    selected_option
)

selected_product = (
    search_result.iloc[selected_idx]
)

# ---------------------------------------
# 선택 상품 정보
# ---------------------------------------
product_name = (
    selected_product['product_name']
)

product_category = (
    selected_product['category']
)

product_price = (
    selected_product['discounted_price']
)

product_discount = (
    selected_product['discount_percentage']
)

product_rating = (
    selected_product['rating']
)

product_review = (
    selected_product['rating_count']
)

product_score = (
    selected_product['consumer_score']
)

# ---------------------------------------
# 판매 중 상품 입력
# ---------------------------------------
if product_type == "현재 판매 중 상품":

    st.sidebar.subheader("📈 현재 판매 상태")

    current_discount = st.sidebar.slider(
        "현재 할인율 (%)",
        0,
        90,
        int(product_discount)
    )

    current_rating = st.sidebar.slider(
        "현재 평점",
        1.0,
        5.0,
        float(product_rating)
    )

    current_review = st.sidebar.number_input(
        "현재 리뷰 수",
        min_value=0,
        value=int(product_review)
    )

# ---------------------------------------
# 메인 상품 정보
# ---------------------------------------
st.header("📦 선택 상품 정보")

col1, col2 = st.columns(2)

with col1:

    st.subheader("상품명")
    st.write(product_name)

    st.subheader("카테고리")
    st.write(product_category)

with col2:

    st.subheader("현재 판매 가격")
    st.write(f"₹ {product_price:.0f}")

    st.subheader("현재 할인율")
    st.write(f"{product_discount:.1f}%")

# ---------------------------------------
# 경쟁 상품 분석
# ---------------------------------------
competitors = df[
    df['category'] == product_category
]

# 비슷한 가격대
price_min = product_price * 0.7
price_max = product_price * 1.3

competitors = competitors[
    (competitors['discounted_price'] >= price_min) &
    (competitors['discounted_price'] <= price_max)
]

# 소비자 반응 순 정렬
competitors = competitors.sort_values(
    by='consumer_score',
    ascending=False
)

top_competitors = competitors.head(10)

# ---------------------------------------
# 시장 평균 계산
# ---------------------------------------
market_discount = (
    top_competitors['discount_percentage']
    .mean()
)

market_rating = (
    top_competitors['rating']
    .mean()
)

market_review = (
    top_competitors['rating_count']
    .mean()
)

market_score = (
    top_competitors['consumer_score']
    .mean()
)

# ---------------------------------------
# AI 시장 분석
# ---------------------------------------
st.header("📊 AI 시장 분석")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "시장 평균 할인율",
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

c4.metric(
    "경쟁 상품 수",
    len(competitors)
)

# ---------------------------------------
# 가격 포지션 분석
# ---------------------------------------
st.subheader("🧠 가격 포지션 분석")

if product_price < competitors[
    'discounted_price'
].quantile(0.33):

    position = "저가형 시장"

elif product_price < competitors[
    'discounted_price'
].quantile(0.66):

    position = "중간 가격 시장"

else:

    position = "프리미엄 시장"

st.info(
    f"현재 상품은 **{position}** 에 위치합니다."
)

# ---------------------------------------
# 경쟁 강도 분석
# ---------------------------------------
st.subheader("⚔️ 경쟁 강도 분석")

competition_level = len(competitors)

if competition_level >= 100:

    st.error("""
경쟁 강도가 매우 높은 시장입니다.

가격 경쟁 및 할인 전략이 중요합니다.
""")

elif competition_level >= 40:

    st.warning("""
중간 수준 경쟁 시장입니다.

리뷰 확보 전략이 중요합니다.
""")

else:

    st.success("""
경쟁 강도가 비교적 낮습니다.

신규 진입 가능성이 있습니다.
""")

# ---------------------------------------
# AI 할인 전략 추천
# ---------------------------------------
st.subheader("💡 AI 할인 전략 추천")

recommended_min = market_discount - 5
recommended_max = market_discount + 5

st.success(f"""
추천 할인율 범위:
### {recommended_min:.1f}% ~ {recommended_max:.1f}%

비슷한 가격대와 카테고리 상품들은
평균적으로 약 **{market_discount:.1f}% 할인율**에서
높은 소비자 반응을 보였습니다.
""")

# ---------------------------------------
# 판매 중 상품 전략 평가
# ---------------------------------------
if product_type == "현재 판매 중 상품":

    st.subheader("📈 현재 판매 전략 평가")

    current_score = (
        current_rating *
        np.log1p(current_review)
    )

    if current_score > market_score:

        st.success("""
현재 상품은 시장 평균보다
높은 소비자 반응을 보이고 있습니다.
""")

    else:

        st.warning("""
현재 상품 반응이 시장 평균 대비 낮습니다.

할인율 조정 또는 리뷰 확보 전략이 필요할 수 있습니다.
""")

    # 할인 전략 비교
    if current_discount < market_discount:

        st.info(f"""
현재 할인율이 시장 평균보다 낮습니다.

추천:
{current_discount}% → {market_discount:.1f}%
수준 검토
""")

    elif current_discount > market_discount + 10:

        st.warning("""
현재 할인율이 다소 높은 편입니다.

수익성 개선 가능성이 있습니다.
""")

# ---------------------------------------
# 경쟁 상품 TOP10
# ---------------------------------------
st.header("🏆 경쟁 상품 TOP10")

show_cols = [
    'product_name',
    'discounted_price',
    'discount_percentage',
    'rating',
    'rating_count'
]

st.dataframe(
    top_competitors[show_cols]
)

# ---------------------------------------
# 할인율 그래프
# ---------------------------------------
st.subheader("📉 경쟁 상품 할인율 비교")

discount_chart = top_competitors[
    ['product_name',
     'discount_percentage']
]

discount_chart = discount_chart.set_index(
    'product_name'
)

st.bar_chart(discount_chart)

# ---------------------------------------
# 리뷰 수 그래프
# ---------------------------------------
st.subheader("⭐ 경쟁 상품 리뷰 수 비교")

review_chart = top_competitors[
    ['product_name',
     'rating_count']
]

review_chart = review_chart.set_index(
    'product_name'
)

st.bar_chart(review_chart)

# ---------------------------------------
# 전체 데이터 보기
# ---------------------------------------
with st.expander("전체 데이터 보기"):

    st.dataframe(df.head(100))

st.markdown("""
<style>

/* 전체 배경 */
.stApp {
    background-color: #FFF9F0;
    color: #4E3D35;
}

/* 메인 영역 */
.main .block-container {
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* 사이드바 */
section[data-testid="stSidebar"] {
    background-color: #FFD98E;
    border-right: 3px solid #F6C667;
}

/* 사이드바 글씨 */
section[data-testid="stSidebar"] * {
    color: #5B4636 !important;
}

/* 입력창 */
.stTextInput input {
    background-color: white !important;
    color: #4E3D35 !important;
    border-radius: 15px !important;
    border: 2px solid #F6C667 !important;
}

/* selectbox */
.stSelectbox div[data-baseweb="select"] {
    background-color: white !important;
    border-radius: 15px !important;
}

/* number input */
.stNumberInput input {
    background-color: white !important;
    color: #4E3D35 !important;
}

/* metric 카드 */
div[data-testid="metric-container"] {
    background-color: white;
    border-radius: 25px;
    padding: 20px;
    border: 2px solid #FFE4B5;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
}

/* 제목 */
h1 {
    color: #5B4636;
    font-size: 3rem;
}

/* 소제목 */
h2, h3 {
    color: #7A5C58;
}

/* dataframe */
[data-testid="stDataFrame"] {
    border-radius: 20px;
    overflow: hidden;
}

/* 버튼 */
.stButton button {
    background-color: #FFB86B !important;
    color: white !important;
    border-radius: 15px !important;
    border: none !important;
    font-weight: bold;
}

/* expander */
.streamlit-expanderHeader {
    background-color: #FFF1D6;
    border-radius: 15px;
}

/* radio */
.stRadio label {
    font-weight: 600;
}

/* slider */
.stSlider {
    color: #F4A261 !important;
}

</style>
""", unsafe_allow_html=True)
