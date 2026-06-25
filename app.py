import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="Seller Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background-color: #F6F4FF;
    font-family: 'Inter', 'Trebuchet MS', sans-serif;
    color: #2D2B55;
}

.block-container {
    padding-top: 2rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
    max-width: 1400px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #E8E3FF 0%, #DCD6FF 100%);
    border-right: 3px solid #C5BDFF;
}

section[data-testid="stSidebar"] * {
    color: #3A3868 !important;
}

h1 { color: #4C44C6; font-size: 2.6rem; font-weight: 800; letter-spacing: -0.5px; }
h2 { color: #5B52CC; font-size: 1.5rem; font-weight: 700; }
h3 { color: #6B64D4; font-size: 1.15rem; font-weight: 600; }

div[data-testid="metric-container"] {
    background: white;
    border-radius: 20px;
    padding: 20px;
    border: 2px solid #DDD8FF;
    box-shadow: 0 4px 16px rgba(100,90,220,0.07);
}

div[data-testid="metric-container"] label {
    color: #7B75CC !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stTextInput input, .stNumberInput input {
    background-color: white !important;
    border-radius: 14px !important;
    border: 2px solid #C5BDFF !important;
    color: #2D2B55 !important;
    font-size: 15px !important;
    padding: 10px 14px !important;
}

.stSelectbox div[data-baseweb="select"] {
    background-color: white !important;
    border-radius: 14px !important;
    border: 2px solid #C5BDFF !important;
}

.stButton button {
    background: linear-gradient(135deg, #7B74E8, #5C54D4) !important;
    color: white !important;
    border-radius: 14px !important;
    border: none !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 12px 24px !important;
    width: 100%;
    transition: opacity 0.2s;
}

.stButton button:hover { opacity: 0.88; }

.card {
    background: white;
    padding: 24px 28px;
    border-radius: 22px;
    border: 2px solid #E0DBFF;
    box-shadow: 0 4px 18px rgba(100,90,220,0.07);
    margin-bottom: 18px;
}

.card-accent {
    background: linear-gradient(135deg, #EEEAFF, #F6F4FF);
    padding: 24px 28px;
    border-radius: 22px;
    border: 2px solid #C5BDFF;
    box-shadow: 0 4px 18px rgba(100,90,220,0.09);
    margin-bottom: 18px;
}

.tag {
    display: inline-block;
    background: #EEEAFF;
    color: #5B52CC;
    border-radius: 999px;
    padding: 3px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 6px;
}

.strategy-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #4C44C6;
    margin-bottom: 6px;
}

.strategy-body {
    font-size: 0.97rem;
    color: #4A4870;
    line-height: 1.7;
}

.highlight-number {
    font-size: 2.4rem;
    font-weight: 800;
    color: #4C44C6;
    letter-spacing: -1px;
}

.section-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #9B96D8;
    margin-bottom: 10px;
}

[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    border: 2px solid #E0DBFF;
}

.stAlert {
    border-radius: 16px !important;
}

div[data-testid="stHorizontalBlock"] > div {
    gap: 16px;
}

.model-badge {
    background: #EEF0FF;
    border: 1.5px solid #C5BDFF;
    border-radius: 10px;
    padding: 6px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #5B52CC;
    display: inline-block;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 데이터 + 모델 캐싱
# =========================================================
@st.cache_data(show_spinner=False)
def load_and_preprocess():
    df = pd.read_csv("amazon.csv")

    def clean_col(s, chars):
        result = s.astype(str)
        for c in chars:
            result = result.str.replace(c, '', regex=False)
        return result

    df['actual_price']        = pd.to_numeric(clean_col(df['actual_price'], ['₹', ',']), errors='coerce')
    df['discounted_price']    = pd.to_numeric(clean_col(df['discounted_price'], ['₹', ',']), errors='coerce')
    df['discount_percentage'] = pd.to_numeric(clean_col(df['discount_percentage'], ['%']), errors='coerce')
    df['rating']              = pd.to_numeric(df['rating'], errors='coerce')
    df['rating_count']        = pd.to_numeric(clean_col(df['rating_count'], [',']), errors='coerce')

    df['main_category'] = df['category'].astype(str).str.split('|').str[0]

    cat_map = {
        "Computers&Accessories": "💻 Computer",
        "Electronics": "🎧 Electronics",
        "Home&Kitchen": "🏠 Home",
        "OfficeProducts": "🖨 Office",
        "MusicalInstruments": "🎵 Music",
        "Health&PersonalCare": "💖 Health",
        "HomeImprovement": "🛠 Tools",
        "Toys&Games": "🧸 Toys"
    }
    df['display_category'] = df['main_category'].map(cat_map).fillna("✨ Other")

    df = df.dropna(subset=[
        'product_name', 'actual_price', 'discount_percentage',
        'rating', 'rating_count', 'display_category',
        'about_product', 'review_title', 'review_content'
    ]).reset_index(drop=True)

    # price_level (카테고리 내 상대 가격대)
    def price_level(row):
        prices = df.loc[df['main_category'] == row['main_category'], 'actual_price']
        q1, q2 = prices.quantile(0.33), prices.quantile(0.66)
        if row['actual_price'] <= q1:   return 'Low'
        elif row['actual_price'] <= q2: return 'Medium'
        else:                           return 'High'

    df['price_level'] = df.apply(price_level, axis=1)
    df['log_rating_count'] = np.log1p(df['rating_count'])
    df['price_discount_interaction'] = df['actual_price'] * df['discount_percentage']
    df['consumer_score'] = df['rating'] * np.log1p(df['rating_count'])
    df['combined_text'] = (
        df['product_name'].fillna('') + ' ' +
        df['about_product'].fillna('') + ' ' +
        df['display_category'].fillna('')
    )

    return df


@st.cache_resource(show_spinner=False)
def train_models(df):
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 모델 1: log 리뷰 수 예측
    feat_review = ['actual_price', 'discount_percentage', 'price_discount_interaction', 'main_category', 'price_level', 'rating']
    num_r = ['actual_price', 'discount_percentage', 'price_discount_interaction', 'rating']
    cat_r = ['main_category', 'price_level']

    pre_r = ColumnTransformer([('num', numeric_pipe, num_r), ('cat', categorical_pipe, cat_r)])
    model_r = Pipeline([('pre', pre_r), ('reg', GradientBoostingRegressor(n_estimators=100, random_state=42))])

    X_r = df[feat_review]
    y_r = df['log_rating_count']
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
    model_r.fit(Xr_tr, yr_tr)
    r2_r = r2_score(yr_te, model_r.predict(Xr_te))

    # 모델 2: 평점 예측
    feat_rating = ['actual_price', 'discount_percentage', 'price_discount_interaction', 'main_category', 'price_level']
    num_rt = ['actual_price', 'discount_percentage', 'price_discount_interaction']
    cat_rt = ['main_category', 'price_level']

    pre_rt = ColumnTransformer([('num', numeric_pipe, num_rt), ('cat', categorical_pipe, cat_rt)])
    model_rt = Pipeline([('pre', pre_rt), ('reg', GradientBoostingRegressor(n_estimators=100, random_state=42))])

    X_rt = df[feat_rating]
    y_rt = df['rating']
    Xrt_tr, Xrt_te, yrt_tr, yrt_te = train_test_split(X_rt, y_rt, test_size=0.2, random_state=42)
    model_rt.fit(Xrt_tr, yrt_tr)
    r2_rt = r2_score(yrt_te, model_rt.predict(Xrt_te))

    return model_r, model_rt, round(r2_r, 3), round(r2_rt, 3)


@st.cache_resource(show_spinner=False)
def build_tfidf(df):
    vec = TfidfVectorizer(stop_words='english', max_features=8000)
    mat = vec.fit_transform(df['combined_text'])
    return vec, mat


# =========================================================
# ABSA
# =========================================================
ASPECT_KEYWORDS = {
    '가격/가성비': ['price', 'cheap', 'expensive', 'value', 'worth', 'budget', 'cost'],
    '품질':       ['quality', 'durable', 'material', 'build', 'sturdy'],
    '배송/포장':  ['delivery', 'shipping', 'package', 'packing', 'arrived'],
    '사용성':     ['easy', 'comfortable', 'convenient', 'install', 'simple'],
    '디자인':     ['design', 'look', 'style', 'color', 'appearance'],
    '성능':       ['performance', 'fast', 'battery', 'sound', 'noise', 'power', 'speed']
}
POSITIVE_WORDS = ['good','great','excellent','amazing','fast','love','perfect','easy',
                  'comfortable','nice','best','awesome','satisfied','worth','happy']
NEGATIVE_WORDS = ['bad','poor','terrible','worst','slow','hate','broken','problem',
                  'difficult','hard','noise','damaged','waste','issue','disappoint']

def run_absa(products_df):
    rows = []
    for _, row in products_df.iterrows():
        text = (str(row['review_title']) + ' ' + str(row['review_content'])).lower().split()
        for aspect, keywords in ASPECT_KEYWORDS.items():
            pos, neg = 0, 0
            for kw in keywords:
                for i, w in enumerate(text):
                    if kw in w:
                        window = text[max(0, i-3): min(len(text), i+4)]
                        for nw in window:
                            if nw in POSITIVE_WORDS: pos += 1
                            elif nw in NEGATIVE_WORDS: neg += 1
            total = pos + neg
            if total > 0:
                sentiment = '긍정' if pos > neg else ('부정' if neg > pos else '중립')
                rows.append([aspect, sentiment, total])

    if not rows:
        return pd.DataFrame(columns=['속성','감정','언급 횟수'])
    return pd.DataFrame(rows, columns=['속성','감정','언급 횟수'])


# =========================================================
# 할인율별 시뮬레이션
# =========================================================
def simulate_discounts(model_r, model_rt, actual_price, main_category, price_level):
    discount_range = list(range(0, 51, 5))
    results = []
    for d in discount_range:
        interaction = actual_price * d
        rt_input = pd.DataFrame({
            'actual_price': [actual_price],
            'discount_percentage': [d],
            'price_discount_interaction': [interaction],
            'main_category': [main_category],
            'price_level': [price_level]
        })
        pred_rating = float(np.clip(model_rt.predict(rt_input)[0], 1, 5))

        rv_input = pd.DataFrame({
            'actual_price': [actual_price],
            'discount_percentage': [d],
            'price_discount_interaction': [interaction],
            'main_category': [main_category],
            'price_level': [price_level],
            'rating': [pred_rating]
        })
        pred_log_rv = model_r.predict(rv_input)[0]
        pred_rv = max(0, int(np.expm1(pred_log_rv)))
        results.append([d, int(actual_price * (1 - d/100)), round(pred_rating, 2), pred_rv])

    sim = pd.DataFrame(results, columns=['할인율(%)', '할인 후 가격', '예상 평점', '예상 리뷰 수'])
    max_rv = sim['예상 리뷰 수'].max()
    min_rv = sim['예상 리뷰 수'].min()

    sim['리뷰수_정규화'] = sim['예상 리뷰 수'] / (max_rv + 1e-9)
    sim['평점_정규화']   = sim['예상 평점'] / 5
    sim['예상매출지표']  = sim['할인 후 가격'] * sim['예상 리뷰 수']
    sim['매출_정규화']   = sim['예상매출지표'] / (sim['예상매출지표'].max() + 1e-9)
    sim['추천점수']      = sim['평점_정규화'] * 0.3 + sim['리뷰수_정규화'] * 0.3 + sim['매출_정규화'] * 0.4
    return sim


# =========================================================
# 전략 코멘트
# =========================================================
ASPECT_COMMENTS = {
    '가격/가성비': ('가성비 강조 전략', '유사 상품 리뷰에서 가격 대비 만족도 언급이 가장 많았습니다. 상세페이지에서 가성비와 할인 혜택을 전면에 내세우는 것이 효과적입니다.'),
    '품질':       ('품질 신뢰성 전략', '소비자들이 내구성·완성도를 중점적으로 언급했습니다. 인증 자료나 소재 정보를 상세페이지 상단에 배치하세요.'),
    '배송/포장':  ('물류 경쟁력 전략', '배송 속도·포장 상태에 대한 반응이 높았습니다. 빠른 배송과 안전 포장을 마케팅 핵심 메시지로 활용하세요.'),
    '사용성':     ('편의성 소구 전략', '사용 편의성과 직관적 조작법에 대한 언급이 많았습니다. 사용 가이드 영상이나 간편 설치 포인트를 강조하세요.'),
    '디자인':     ('비주얼 마케팅 전략', '디자인·외형 관련 피드백이 높게 나타났습니다. 감성적인 제품 연출 이미지와 색상 다양성을 전면에 배치하세요.'),
    '성능':       ('스펙 중심 전략', '성능 관련 언급이 가장 강했습니다. 배터리, 속도, 출력 등 핵심 스펙을 수치와 함께 소구하는 전략이 유리합니다.')
}

def get_discount_strategy(best_discount):
    if best_discount <= 10:
        return ('수익성 유지형', '낮은 할인율에서도 평점·리뷰가 안정적으로 유지됩니다. 과도한 할인보다 마진 방어에 집중하는 것이 유리합니다.')
    elif best_discount <= 20:
        return ('균형형', '중간 할인율에서 리뷰 수·평점·매출의 밸런스가 최적화됩니다. 가격 경쟁력과 마진을 동시에 확보하는 전략입니다.')
    else:
        return ('공격형 프로모션', '높은 할인율에서 리뷰 유입 효과가 크게 나타납니다. 단기 노출 확대와 리뷰 대량 확보를 목표로 한 전략에 적합합니다.')


# =========================================================
# Plotly 차트 헬퍼
# =========================================================
COLORS = {
    'main': '#7B74E8', 'point': '#5C54D4', 'highlight': '#FF7E7E',
    'sub': '#B5B0F5', 'bg': '#F6F4FF', 'text': '#2D2B55', 'grid': '#E8E4FF'
}

def make_line_chart(x, y, title, xlab, ylab, best_x=None, y_fmt=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines+markers+text',
        line=dict(color=COLORS['main'], width=3),
        marker=dict(size=9, color=COLORS['main'], line=dict(color='white', width=2)),
        text=[f"{v:,}" if y_fmt == 'int' else f"{v:.2f}" for v in y],
        textposition='top center', textfont=dict(size=11, color=COLORS['text'])
    ))
    if best_x is not None:
        fig.add_vline(x=best_x, line_dash='dash', line_color=COLORS['highlight'], line_width=2.5,
                      annotation_text=f"추천 {best_x}%", annotation_font_color=COLORS['highlight'],
                      annotation_font_size=13)
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=COLORS['text'], weight=700)),
        xaxis=dict(title=xlab, gridcolor=COLORS['grid'], showgrid=True),
        yaxis=dict(title=ylab, gridcolor=COLORS['grid'], showgrid=True),
        paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(l=20, r=20, t=50, b=30), height=320
    )
    return fig

def make_bar_chart(x, y, title, best_x=None):
    colors = [COLORS['highlight'] if xi == best_x else COLORS['sub'] for xi in x]
    fig = go.Figure(go.Bar(
        x=[str(v) for v in x], y=y, marker_color=colors,
        text=[f"{v:.3f}" for v in y], textposition='outside', textfont=dict(size=10, color=COLORS['text'])
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=COLORS['text'], weight=700)),
        xaxis=dict(title='할인율(%)', gridcolor=COLORS['grid']),
        yaxis=dict(title='추천점수', gridcolor=COLORS['grid'], showgrid=True),
        paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(l=20, r=20, t=50, b=30), height=300
    )
    return fig

def make_absa_chart(absa_df):
    if absa_df.empty:
        return None
    pivot = absa_df.groupby(['속성','감정'])['언급 횟수'].sum().reset_index()
    color_map = {'긍정': COLORS['main'], '중립': '#C5BDFF', '부정': COLORS['highlight']}
    fig = px.bar(pivot, x='속성', y='언급 횟수', color='감정', barmode='stack',
                 color_discrete_map=color_map, title='속성별 소비자 반응 (ABSA)')
    fig.update_layout(
        paper_bgcolor='white', plot_bgcolor='white',
        xaxis=dict(gridcolor=COLORS['grid']), yaxis=dict(gridcolor=COLORS['grid'], showgrid=True),
        margin=dict(l=20, r=20, t=50, b=30), height=320,
        title_font=dict(size=15, color=COLORS['text'], weight=700)
    )
    return fig

def make_market_scatter(df, input_price, pred_log_rv):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['actual_price'], y=df['log_rating_count'],
        mode='markers',
        marker=dict(size=6, color=df['rating'], colorscale='Blues',
                    opacity=0.55, showscale=True,
                    colorbar=dict(title='평점', thickness=12)),
        hovertemplate='가격: %{x:,}<br>log(리뷰 수): %{y:.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=[input_price], y=[pred_log_rv],
        mode='markers',
        marker=dict(size=22, symbol='star', color=COLORS['highlight'],
                    line=dict(color='white', width=2)),
        name='선택 상품 예상 위치',
        hovertemplate='선택 상품<br>가격: %{x:,}<br>log(리뷰 수): %{y:.2f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text='시장 내 상품 위치', font=dict(size=15, color=COLORS['text'], weight=700)),
        xaxis=dict(title='가격', gridcolor=COLORS['grid'], showgrid=True),
        yaxis=dict(title='log(리뷰 수 + 1)', gridcolor=COLORS['grid'], showgrid=True),
        paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(l=20, r=20, t=50, b=30), height=340,
        showlegend=True
    )
    return fig


# =========================================================
# 메인
# =========================================================
with st.spinner("📦 데이터 로딩 중..."):
    df = load_and_preprocess()

with st.spinner("🤖 AI 모델 학습 중 (최초 1회)..."):
    model_r, model_rt, r2_review, r2_rating = train_models(df)
    vectorizer, tfidf_matrix = build_tfidf(df)

# =========================================================
# 헤더
# =========================================================
st.title("🛒 Seller Assistant")
st.markdown("""
<p style="font-size:1.05rem; color:#6B64D4; margin-top:-10px; margin-bottom:24px;">
AI 기반 아마존 판매 전략 분석 도구 &nbsp;·&nbsp; 할인율 시뮬레이션 &nbsp;·&nbsp; 소비자 반응 분석
</p>
""", unsafe_allow_html=True)

# 모델 상태 배지
c_m1, c_m2, c_info = st.columns([1, 1, 3])
c_m1.markdown(f'<div class="model-badge">🤖 리뷰 예측 모델 &nbsp; R² = {r2_review}</div>', unsafe_allow_html=True)
c_m2.markdown(f'<div class="model-badge">⭐ 평점 예측 모델 &nbsp; R² = {r2_rating}</div>', unsafe_allow_html=True)

st.divider()

# =========================================================
# 사이드바
# =========================================================
st.sidebar.header("🎮 분석 설정")

product_type = st.sidebar.radio("상품 유형", ["판매 중 상품", "신상품 (직접 입력)"])

if product_type == "판매 중 상품":
    st.sidebar.subheader("🔍 상품 검색")
    recommended_items = ["earbuds","headphones","keyboard","mouse","tablet",
                         "speaker","charger","smartwatch","laptop","cable"]
    sel_item = st.sidebar.selectbox("추천 검색어", recommended_items)
    search_kw = st.sidebar.text_input("상품 검색", sel_item)

    search_result = df[df['product_name'].str.contains(search_kw, case=False, na=False)].head(20).reset_index(drop=True)
    options = [f"[{i+1}] {r['product_name'][:60]}" for i, r in search_result.iterrows()]

    if not options:
        st.sidebar.warning("검색 결과가 없습니다.")
        st.stop()

    sel_opt = st.sidebar.selectbox("분석 상품 선택", options)
    sel_idx = options.index(sel_opt)
    sel_row = search_result.iloc[sel_idx]

    product_name       = sel_row['product_name']
    main_category      = sel_row['main_category']
    display_category   = sel_row['display_category']
    actual_price       = float(sel_row['actual_price'])
    price_level_input  = sel_row['price_level']
    about_product      = sel_row.get('about_product', '')

else:
    st.sidebar.subheader("✨ 신상품 정보 입력")
    product_name = st.sidebar.text_input("상품명", "Wireless Headset")
    about_product = st.sidebar.text_input("설명 키워드", "bluetooth noise cancelling")
    category_list = sorted(df['display_category'].dropna().unique())
    display_category = st.sidebar.selectbox("카테고리", category_list)
    main_category = df.loc[df['display_category'] == display_category, 'main_category'].iloc[0]
    actual_price = st.sidebar.number_input("예상 판매 가격 (₹)", min_value=0.0, value=5000.0)
    # 가격 레벨 자동 산정
    prices = df.loc[df['main_category'] == main_category, 'actual_price']
    q1, q2 = prices.quantile(0.33), prices.quantile(0.66)
    price_level_input = 'Low' if actual_price <= q1 else ('Medium' if actual_price <= q2 else 'High')

st.sidebar.divider()
analyze_button = st.sidebar.button("🎀 분석 시작", use_container_width=True)


# =========================================================
# 대기 화면
# =========================================================
if not analyze_button:
    st.markdown("""
    <div class="card" style="text-align:center; padding: 60px 40px;">
        <div style="font-size:3.5rem; margin-bottom:16px;">🛒</div>
        <h2 style="color:#4C44C6; margin-bottom:10px;">왼쪽에서 상품을 선택하세요</h2>
        <p style="color:#8880C8; font-size:1rem;">
            판매 중인 상품 또는 신상품 정보를 입력한 뒤<br>
            <b>🎀 분석 시작</b> 버튼을 누르면 AI 분석 결과가 나타납니다.
        </p>
        <br>
        <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap; margin-top:8px;">
            <span class="tag">📊 할인율 시뮬레이션</span>
            <span class="tag">🤖 AI 평점·리뷰 예측</span>
            <span class="tag">🔍 유사 상품 분석</span>
            <span class="tag">💬 소비자 반응 (ABSA)</span>
            <span class="tag">💡 판매 전략 추천</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# =========================================================
# 분석 실행
# =========================================================
with st.spinner("🤖 AI 분석 중..."):

    # 1) 할인율 시뮬레이션
    sim_df = simulate_discounts(model_r, model_rt, actual_price, main_category, price_level_input)
    best = sim_df.sort_values('추천점수', ascending=False).iloc[0]

    # 2) 유사 상품 TF-IDF
    user_text = product_name + ' ' + about_product + ' ' + display_category
    user_vec = vectorizer.transform([user_text])
    scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    df['similarity_score'] = scores

    similar = (
        df[df['product_name'] != product_name]
        .sort_values('similarity_score', ascending=False)
        .head(15)
        .copy()
    )

    # 3) ABSA
    absa_raw = run_absa(similar)
    absa_summary = (
        absa_raw.groupby(['속성','감정'])['언급 횟수'].sum().reset_index()
        if not absa_raw.empty else absa_raw
    )

    # 4) 상위 속성
    kw_rows = []
    all_text = ' '.join(
        (similar['review_title'].astype(str) + ' ' + similar['review_content'].astype(str)).tolist()
    ).lower()
    for aspect, kws in ASPECT_KEYWORDS.items():
        kw_rows.append([aspect, sum(all_text.count(kw) for kw in kws)])
    kw_df = pd.DataFrame(kw_rows, columns=['속성','언급 횟수']).sort_values('언급 횟수', ascending=False)
    top_aspect = kw_df.iloc[0]['속성']

    # 5) 전략 코멘트
    discount_strat_name, discount_strat_body = get_discount_strategy(int(best['할인율(%)']))
    aspect_title, aspect_body = ASPECT_COMMENTS.get(top_aspect, (top_aspect, ''))

    # 6) 시장 통계
    price_pct  = (df['actual_price'] < actual_price).mean() * 100
    review_pct = (df['rating_count'] < best['예상 리뷰 수']).mean() * 100
    rating_pct = (df['rating'] < best['예상 평점']).mean() * 100
    competition = len(df[df['display_category'] == display_category])


# =========================================================
# ── 결과 출력 ──
# =========================================================

# ── 상품 정보 카드
st.markdown(f"""
<div class="card">
  <div class="section-label">분석 대상</div>
  <div style="font-size:1.2rem; font-weight:700; color:#2D2B55; margin-bottom:10px;">
    {product_name[:90]}
  </div>
  <span class="tag">{display_category}</span>
  <span class="tag">₹ {int(actual_price):,}</span>
  <span class="tag">가격대: {price_level_input}</span>
  <span class="tag">{product_type}</span>
</div>
""", unsafe_allow_html=True)


# ── 핵심 지표
st.markdown("### 📊 AI 예측 핵심 지표")
m1, m2, m3, m4 = st.columns(4)
m1.metric("추천 할인율",      f"{int(best['할인율(%)'])}%")
m2.metric("예상 평점",         f"{best['예상 평점']:.2f} / 5")
m3.metric("예상 리뷰 수",      f"{best['예상 리뷰 수']:,}개")
m4.metric("시장 경쟁 상품 수", f"{competition:,}개")

st.divider()


# ── 차트 섹션
st.markdown("### 📈 할인율별 예측 시뮬레이션")
col_l, col_r = st.columns(2)
with col_l:
    st.plotly_chart(
        make_line_chart(sim_df['할인율(%)'], sim_df['예상 리뷰 수'],
                        '할인율별 예상 리뷰 수', '할인율(%)', '예상 리뷰 수',
                        best_x=int(best['할인율(%)']), y_fmt='int'),
        use_container_width=True
    )
with col_r:
    st.plotly_chart(
        make_line_chart(sim_df['할인율(%)'], sim_df['예상 평점'],
                        '할인율별 예상 평점', '할인율(%)', '예상 평점',
                        best_x=int(best['할인율(%)'])),
        use_container_width=True
    )

col_bar, col_scatter = st.columns(2)
with col_bar:
    st.plotly_chart(
        make_bar_chart(sim_df['할인율(%)'], sim_df['추천점수'],
                       '할인율별 추천점수 (평점 30% + 리뷰 30% + 매출 40%)',
                       best_x=int(best['할인율(%)'])),
        use_container_width=True
    )
with col_scatter:
    best_log_rv = np.log1p(best['예상 리뷰 수'])
    st.plotly_chart(
        make_market_scatter(df, actual_price, best_log_rv),
        use_container_width=True
    )

st.divider()


# ── ABSA
st.markdown("### 💬 소비자 반응 분석 (ABSA)")
absa_col, kw_col = st.columns([3, 2])
with absa_col:
    absa_fig = make_absa_chart(absa_summary)
    if absa_fig:
        st.plotly_chart(absa_fig, use_container_width=True)
    else:
        st.info("ABSA 분석에 충분한 리뷰 텍스트가 없습니다.")
with kw_col:
    st.markdown("**속성별 언급 빈도**")
    kw_fig = px.bar(
        kw_df.sort_values('언급 횟수'),
        x='언급 횟수', y='속성', orientation='h',
        color_discrete_sequence=[COLORS['sub']]
    )
    kw_fig.update_layout(
        paper_bgcolor='white', plot_bgcolor='white', height=320,
        margin=dict(l=10, r=20, t=20, b=20),
        xaxis=dict(gridcolor=COLORS['grid'], showgrid=True),
        yaxis=dict(gridcolor=COLORS['grid'])
    )
    st.plotly_chart(kw_fig, use_container_width=True)

st.divider()


# ── 유사 상품
st.markdown("### 🔍 유사 상품 분석")
with st.expander("유사 상품 TOP 15 보기", expanded=False):
    show_similar = similar[[
        'product_name', 'display_category', 'actual_price',
        'discount_percentage', 'rating', 'rating_count', 'similarity_score'
    ]].copy()
    show_similar.columns = ['상품명','카테고리','가격','할인율(%)','평점','리뷰 수','유사도']
    show_similar = show_similar.reset_index(drop=True)
    st.dataframe(show_similar, use_container_width=True)

st.divider()


# ── 전략 리포트
st.markdown("### 💡 AI 판매 전략 리포트")

s1, s2 = st.columns(2)

with s1:
    st.markdown(f"""
    <div class="card-accent">
      <div class="section-label">할인 전략</div>
      <div class="strategy-title">📌 {int(best['할인율(%)'])}% &nbsp; {discount_strat_name}</div>
      <div class="strategy-body">{discount_strat_body}</div>
    </div>
    """, unsafe_allow_html=True)

with s2:
    st.markdown(f"""
    <div class="card-accent">
      <div class="section-label">소비자 소구 전략</div>
      <div class="strategy-title">💬 {aspect_title}</div>
      <div class="strategy-body">{aspect_body}</div>
    </div>
    """, unsafe_allow_html=True)

# 시장 내 위치
st.markdown(f"""
<div class="card">
  <div class="section-label">시장 내 위치 분석</div>
  <div style="display:flex; gap:32px; flex-wrap:wrap; margin-top:8px;">
    <div>
      <div style="font-size:0.82rem; color:#9B96D8; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">가격 상위</div>
      <div class="highlight-number">{price_pct:.1f}<span style="font-size:1.2rem; font-weight:600;">%</span></div>
    </div>
    <div>
      <div style="font-size:0.82rem; color:#9B96D8; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">예상 리뷰 상위</div>
      <div class="highlight-number">{review_pct:.1f}<span style="font-size:1.2rem; font-weight:600;">%</span></div>
    </div>
    <div>
      <div style="font-size:0.82rem; color:#9B96D8; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">예상 평점 상위</div>
      <div class="highlight-number">{rating_pct:.1f}<span style="font-size:1.2rem; font-weight:600;">%</span></div>
    </div>
  </div>
  <div style="margin-top:16px; font-size:0.95rem; color:#4A4870; line-height:1.7;">
    현재 분석 기준으로 추천 할인율 <b>{int(best['할인율(%)'])}%</b> 적용 시 
    예상 평점 <b>{best['예상 평점']:.2f}점</b>, 예상 리뷰 <b>{best['예상 리뷰 수']:,}개</b>로 예측됩니다.
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# ── 할인율별 전체 시뮬레이션 테이블
with st.expander("📋 할인율별 전체 예측 결과 테이블"):
    st.dataframe(sim_df[['할인율(%)','할인 후 가격','예상 평점','예상 리뷰 수','추천점수']].reset_index(drop=True),
                 use_container_width=True)
