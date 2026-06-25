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
# CSS — 밝고 모던한 스타트업 스타일
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

/* ── 전체 배경 */
.stApp {
    background-color: #F8F9FC;
    font-family: 'Inter', sans-serif;
    color: #1A1D2E;
}

.block-container {
    padding-top: 2rem;
    padding-left: 2.8rem;
    padding-right: 2.8rem;
    max-width: 1440px;
}

/* ── 사이드바 — 흰 배경 + 좌측 컬러 보더 */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E8EAF2;
    box-shadow: 2px 0 12px rgba(0,0,0,0.04);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.6rem;
    padding-left: 1.4rem;
    padding-right: 1.4rem;
}

/* 사이드바 전체 텍스트 */
section[data-testid="stSidebar"] * {
    color: #1A1D2E !important;
}

/* 사이드바 로고 영역 */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 16px 20px;
    border-bottom: 1px solid #F0F1F8;
    margin-bottom: 20px;
}
.sidebar-logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.sidebar-logo-text {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1A1D2E !important;
    letter-spacing: -0.3px;
}

/* 사이드바 섹션 레이블 */
.sidebar-section-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #9CA3AF !important;
    margin: 18px 0 8px;
    padding: 0 2px;
}

/* 사이드바 라디오 커스텀 */
section[data-testid="stSidebar"] .stRadio > div {
    gap: 6px;
}
section[data-testid="stSidebar"] .stRadio label {
    background: #F8F9FC;
    border: 1.5px solid #E8EAF2;
    border-radius: 10px;
    padding: 9px 14px !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    transition: all 0.15s;
    cursor: pointer;
    width: 100%;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: #EEF2FF;
    border-color: #C7D2FE;
}

/* 입력창 */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stNumberInput input {
    background-color: #F8F9FC !important;
    border: 1.5px solid #E2E5F1 !important;
    border-radius: 10px !important;
    color: #1A1D2E !important;
    font-size: 0.9rem !important;
    padding: 9px 12px !important;
    transition: border-color 0.15s;
}
section[data-testid="stSidebar"] input:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

/* 셀렉트박스 */
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
    background-color: #F8F9FC !important;
    border: 1.5px solid #E2E5F1 !important;
    border-radius: 10px !important;
}

/* 분석 버튼 */
section[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    padding: 13px 20px !important;
    width: 100%;
    letter-spacing: 0.2px;
    box-shadow: 0 4px 14px rgba(79,70,229,0.35) !important;
    transition: all 0.2s !important;
    margin-top: 6px;
}
section[data-testid="stSidebar"] .stButton button:hover {
    box-shadow: 0 6px 20px rgba(79,70,229,0.45) !important;
    transform: translateY(-1px);
}

/* ── 메인 헤더 */
h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #1A1D2E !important;
    letter-spacing: -0.8px !important;
    line-height: 1.2 !important;
}
h2 {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #1A1D2E !important;
    letter-spacing: -0.3px !important;
}
h3 {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #374151 !important;
}

/* ── 메트릭 카드 */
div[data-testid="metric-container"] {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 22px 24px !important;
    border: 1px solid #E8EAF2;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    position: relative;
    overflow: hidden;
}
div[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #4F46E5, #7C3AED);
}
div[data-testid="metric-container"] label {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    color: #9CA3AF !important;
}
div[data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    color: #1A1D2E !important;
    letter-spacing: -1px !important;
}

/* ── 커스텀 카드 */
.card {
    background: #FFFFFF;
    padding: 24px 28px;
    border-radius: 16px;
    border: 1px solid #E8EAF2;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-bottom: 16px;
}

.card-highlight {
    background: #FFFFFF;
    padding: 24px 28px;
    border-radius: 16px;
    border: 1px solid #E0E7FF;
    border-left: 4px solid #4F46E5;
    box-shadow: 0 2px 8px rgba(79,70,229,0.06);
    margin-bottom: 16px;
}

.card-success {
    background: #FFFFFF;
    padding: 24px 28px;
    border-radius: 16px;
    border: 1px solid #D1FAE5;
    border-left: 4px solid #10B981;
    box-shadow: 0 2px 8px rgba(16,185,129,0.06);
    margin-bottom: 16px;
}

/* ── 태그 */
.tag {
    display: inline-flex;
    align-items: center;
    background: #EEF2FF;
    color: #4F46E5;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 4px;
    letter-spacing: 0.1px;
}

.tag-gray {
    display: inline-flex;
    align-items: center;
    background: #F3F4F6;
    color: #6B7280;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 4px;
}

/* ── 섹션 레이블 */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #9CA3AF;
    margin-bottom: 8px;
}

/* ── 모델 배지 */
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 8px;
    padding: 5px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #065F46;
}

/* ── 전략 카드 내부 */
.strategy-title {
    font-size: 1rem;
    font-weight: 700;
    color: #1A1D2E;
    margin-bottom: 8px;
    letter-spacing: -0.2px;
}
.strategy-body {
    font-size: 0.9rem;
    color: #4B5563;
    line-height: 1.7;
}

/* ── 큰 숫자 */
.big-number {
    font-size: 2.6rem;
    font-weight: 800;
    color: #4F46E5;
    letter-spacing: -1.5px;
    line-height: 1;
}
.big-number-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #9CA3AF;
    margin-bottom: 6px;
}

/* ── 구분선 */
hr {
    border: none;
    border-top: 1px solid #F0F1F8;
    margin: 20px 0;
}

/* ── 데이터프레임 */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #E8EAF2 !important;
}

/* ── expander */
.streamlit-expanderHeader {
    background: #F8F9FC !important;
    border-radius: 12px !important;
    border: 1px solid #E8EAF2 !important;
    font-weight: 600 !important;
    color: #374151 !important;
}

/* ── 페이지 섹션 헤더 */
.page-section {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 16px;
}
.page-section-icon {
    width: 32px; height: 32px;
    background: #EEF2FF;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px;
}
.page-section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1A1D2E;
    letter-spacing: -0.3px;
}

/* ── 대기 화면 feature 그리드 */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 24px;
}
.feature-item {
    background: #F8F9FC;
    border: 1px solid #E8EAF2;
    border-radius: 12px;
    padding: 20px;
    text-align: left;
}
.feature-item-icon {
    font-size: 1.5rem;
    margin-bottom: 10px;
}
.feature-item-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #1A1D2E;
    margin-bottom: 4px;
}
.feature-item-desc {
    font-size: 0.8rem;
    color: #6B7280;
    line-height: 1.5;
}

/* divider */
.stDivider { border-color: #F0F1F8 !important; }

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
    numeric_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # 모델 1: log 리뷰 수 예측
    feat_r  = ['actual_price', 'discount_percentage', 'price_discount_interaction', 'main_category', 'price_level', 'rating']
    num_r   = ['actual_price', 'discount_percentage', 'price_discount_interaction', 'rating']
    cat_r   = ['main_category', 'price_level']
    pre_r   = ColumnTransformer([('num', numeric_pipe, num_r), ('cat', categorical_pipe, cat_r)])
    model_r = Pipeline([('pre', pre_r), ('reg', GradientBoostingRegressor(n_estimators=100, random_state=42))])
    X_r, y_r = df[feat_r], df['log_rating_count']
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
    model_r.fit(Xr_tr, yr_tr)
    r2_r = r2_score(yr_te, model_r.predict(Xr_te))

    # 모델 2: 평점 예측
    feat_rt  = ['actual_price', 'discount_percentage', 'price_discount_interaction', 'main_category', 'price_level']
    num_rt   = ['actual_price', 'discount_percentage', 'price_discount_interaction']
    cat_rt   = ['main_category', 'price_level']
    pre_rt   = ColumnTransformer([('num', numeric_pipe, num_rt), ('cat', categorical_pipe, cat_rt)])
    model_rt = Pipeline([('pre', pre_rt), ('reg', GradientBoostingRegressor(n_estimators=100, random_state=42))])
    X_rt, y_rt = df[feat_rt], df['rating']
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
    '품질':        ['quality', 'durable', 'material', 'build', 'sturdy'],
    '배송/포장':   ['delivery', 'shipping', 'package', 'packing', 'arrived'],
    '사용성':      ['easy', 'comfortable', 'convenient', 'install', 'simple'],
    '디자인':      ['design', 'look', 'style', 'color', 'appearance'],
    '성능':        ['performance', 'fast', 'battery', 'sound', 'noise', 'power', 'speed']
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
# 할인율 시뮬레이션
# =========================================================
def simulate_discounts(model_r, model_rt, actual_price, main_category, price_level):
    results = []
    for d in range(0, 51, 5):
        interaction = actual_price * d
        rt_in = pd.DataFrame({'actual_price':[actual_price],'discount_percentage':[d],
                               'price_discount_interaction':[interaction],
                               'main_category':[main_category],'price_level':[price_level]})
        pred_rt = float(np.clip(model_rt.predict(rt_in)[0], 1, 5))
        rv_in = pd.DataFrame({'actual_price':[actual_price],'discount_percentage':[d],
                               'price_discount_interaction':[interaction],
                               'main_category':[main_category],'price_level':[price_level],
                               'rating':[pred_rt]})
        pred_rv = max(0, int(np.expm1(model_r.predict(rv_in)[0])))
        results.append([d, int(actual_price*(1-d/100)), round(pred_rt,2), pred_rv])

    sim = pd.DataFrame(results, columns=['할인율(%)','할인 후 가격','예상 평점','예상 리뷰 수'])
    sim['리뷰수_정규화'] = sim['예상 리뷰 수'] / (sim['예상 리뷰 수'].max() + 1e-9)
    sim['평점_정규화']   = sim['예상 평점'] / 5
    sim['예상매출지표']  = sim['할인 후 가격'] * sim['예상 리뷰 수']
    sim['매출_정규화']   = sim['예상매출지표'] / (sim['예상매출지표'].max() + 1e-9)
    sim['추천점수']      = sim['평점_정규화']*0.3 + sim['리뷰수_정규화']*0.3 + sim['매출_정규화']*0.4
    return sim


# =========================================================
# 전략 텍스트
# =========================================================
ASPECT_COMMENTS = {
    '가격/가성비': ('가성비 강조 전략', '유사 상품 리뷰에서 가격 대비 만족도 언급이 가장 많았습니다. 상세페이지에서 가성비와 할인 혜택을 전면에 내세우는 것이 효과적입니다.'),
    '품질':        ('품질 신뢰성 전략', '소비자들이 내구성·완성도를 중점적으로 언급했습니다. 인증 자료나 소재 정보를 상세페이지 상단에 배치하세요.'),
    '배송/포장':   ('물류 경쟁력 전략', '배송 속도·포장 상태에 대한 반응이 높았습니다. 빠른 배송과 안전 포장을 마케팅 핵심 메시지로 활용하세요.'),
    '사용성':      ('편의성 소구 전략', '사용 편의성과 직관적 조작법에 대한 언급이 많았습니다. 사용 가이드 영상이나 간편 설치 포인트를 강조하세요.'),
    '디자인':      ('비주얼 마케팅 전략', '디자인·외형 관련 피드백이 높게 나타났습니다. 감성적인 제품 연출 이미지와 색상 다양성을 전면에 배치하세요.'),
    '성능':        ('스펙 중심 전략', '성능 관련 언급이 가장 강했습니다. 배터리, 속도, 출력 등 핵심 스펙을 수치와 함께 소구하는 전략이 유리합니다.')
}

def get_discount_strategy(d):
    if d <= 10:
        return ('수익성 유지형', '낮은 할인율에서도 평점·리뷰가 안정적으로 유지됩니다. 과도한 할인보다 마진 방어에 집중하는 것이 유리합니다.')
    elif d <= 20:
        return ('균형형', '중간 할인율에서 리뷰 수·평점·매출의 밸런스가 최적화됩니다. 가격 경쟁력과 마진을 동시에 확보하는 전략입니다.')
    else:
        return ('공격형 프로모션', '높은 할인율에서 리뷰 유입 효과가 크게 나타납니다. 단기 노출 확대와 리뷰 대량 확보를 목표로 한 전략에 적합합니다.')


# =========================================================
# 차트 — 라이트 테마
# =========================================================
C = {
    'indigo':  '#4F46E5',
    'violet':  '#7C3AED',
    'red':     '#EF4444',
    'green':   '#10B981',
    'gray':    '#E5E7EB',
    'text':    '#1A1D2E',
    'subtext': '#6B7280',
    'grid':    '#F3F4F6',
    'bg':      '#FFFFFF',
}

CHART_LAYOUT = dict(
    paper_bgcolor=C['bg'], plot_bgcolor=C['bg'],
    font=dict(family='Inter, sans-serif', color=C['text']),
    margin=dict(l=16, r=16, t=48, b=24),
    hoverlabel=dict(bgcolor='white', font_size=13, font_family='Inter'),
)

def line_chart(x, y, title, xlab, ylab, best_x=None, fmt='int'):
    fig = go.Figure()
    # 영역 채우기
    fig.add_trace(go.Scatter(
        x=x, y=y, fill='tozeroy',
        fillcolor='rgba(79,70,229,0.07)',
        line=dict(color=C['indigo'], width=0),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines+markers',
        line=dict(color=C['indigo'], width=2.5),
        marker=dict(size=8, color='white', line=dict(color=C['indigo'], width=2.5)),
        text=[f"{v:,}" if fmt=='int' else f"{v:.2f}" for v in y],
        textposition='top center',
        textfont=dict(size=10, color=C['subtext']),
        hovertemplate=f'<b>%{{x}}%</b><br>{ylab}: %{{y}}<extra></extra>',
        name=ylab
    ))
    if best_x is not None:
        fig.add_vline(x=best_x, line_dash='dot', line_color=C['red'], line_width=1.8,
                      annotation_text=f"  추천 {best_x}%",
                      annotation_font=dict(color=C['red'], size=12, family='Inter'))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, weight=700, color=C['text'])),
        xaxis=dict(title=xlab, gridcolor=C['grid'], showgrid=True, zeroline=False,
                   tickfont=dict(size=11)),
        yaxis=dict(title=ylab, gridcolor=C['grid'], showgrid=True, zeroline=False,
                   tickfont=dict(size=11)),
        showlegend=False, height=300,
        **CHART_LAYOUT
    )
    return fig

def bar_score_chart(x, y, best_x):
    colors = [C['indigo'] if xi == best_x else C['gray'] for xi in x]
    fig = go.Figure(go.Bar(
        x=[str(v) for v in x], y=y,
        marker=dict(color=colors, line=dict(width=0), cornerradius=6),
        text=[f"{v:.3f}" for v in y],
        textposition='outside',
        textfont=dict(size=10, color=C['subtext']),
        hovertemplate='할인율 %{x}%<br>추천점수: %{y:.3f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text='할인율별 종합 추천점수', font=dict(size=14, weight=700, color=C['text'])),
        xaxis=dict(title='할인율(%)', gridcolor=C['grid'], zeroline=False, tickfont=dict(size=11)),
        yaxis=dict(title='추천점수', gridcolor=C['grid'], showgrid=True, zeroline=False,
                   tickfont=dict(size=11)),
        height=300,
        **CHART_LAYOUT
    )
    return fig

def absa_chart(absa_df):
    if absa_df.empty:
        return None
    pivot = absa_df.groupby(['속성','감정'])['언급 횟수'].sum().reset_index()
    color_map = {'긍정': C['indigo'], '중립': '#A5B4FC', '부정': C['red']}
    fig = px.bar(pivot, x='속성', y='언급 횟수', color='감정', barmode='stack',
                 color_discrete_map=color_map)
    fig.update_traces(marker_line_width=0)
    fig.update_layout(
        title=dict(text='속성별 소비자 반응 (ABSA)', font=dict(size=14, weight=700, color=C['text'])),
        xaxis=dict(gridcolor=C['grid'], zeroline=False, tickfont=dict(size=11)),
        yaxis=dict(gridcolor=C['grid'], showgrid=True, zeroline=False, tickfont=dict(size=11)),
        legend=dict(title='', font=dict(size=11), orientation='h', y=1.1),
        height=300,
        **CHART_LAYOUT
    )
    return fig

def kw_chart(kw_df):
    df_s = kw_df.sort_values('언급 횟수')
    fig = go.Figure(go.Bar(
        x=df_s['언급 횟수'], y=df_s['속성'], orientation='h',
        marker=dict(
            color=df_s['언급 횟수'],
            colorscale=[[0,'#E0E7FF'],[1,C['indigo']]],
            line=dict(width=0),
        ),
        hovertemplate='%{y}: %{x}회<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text='속성별 언급 빈도', font=dict(size=14, weight=700, color=C['text'])),
        xaxis=dict(gridcolor=C['grid'], showgrid=True, zeroline=False, tickfont=dict(size=11)),
        yaxis=dict(gridcolor=C['grid'], zeroline=False, tickfont=dict(size=11)),
        height=300,
        **CHART_LAYOUT
    )
    return fig

def scatter_chart(df, input_price, pred_log_rv):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['actual_price'], y=df['log_rating_count'],
        mode='markers',
        marker=dict(size=5, color=df['rating'], colorscale='Blues',
                    opacity=0.4, showscale=True,
                    colorbar=dict(title='평점', thickness=10, len=0.8)),
        hovertemplate='가격: ₹%{x:,}<br>log(리뷰 수): %{y:.2f}<extra></extra>',
        name='전체 상품'
    ))
    fig.add_trace(go.Scatter(
        x=[input_price], y=[pred_log_rv],
        mode='markers',
        marker=dict(size=18, symbol='star', color=C['red'],
                    line=dict(color='white', width=2)),
        name='선택 상품',
        hovertemplate='선택 상품<br>가격: ₹%{x:,}<br>log(리뷰 수): %{y:.2f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text='시장 내 상품 포지션', font=dict(size=14, weight=700, color=C['text'])),
        xaxis=dict(title='가격 (₹)', gridcolor=C['grid'], showgrid=True, zeroline=False),
        yaxis=dict(title='log(리뷰 수 + 1)', gridcolor=C['grid'], showgrid=True, zeroline=False),
        legend=dict(font=dict(size=11), orientation='h', y=1.1),
        height=300,
        **CHART_LAYOUT
    )
    return fig


# =========================================================
# 초기화
# =========================================================
with st.spinner("데이터 로딩 중..."):
    df = load_and_preprocess()
with st.spinner("AI 모델 준비 중 (최초 1회)..."):
    model_r, model_rt, r2_review, r2_rating = train_models(df)
    vectorizer, tfidf_matrix = build_tfidf(df)


# =========================================================
# 사이드바
# =========================================================
with st.sidebar:
    # 로고
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🛒</div>
        <div class="sidebar-logo-text">Seller Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">상품 유형</div>', unsafe_allow_html=True)
    product_type = st.radio("", ["판매 중 상품", "신상품 (직접 입력)"], label_visibility="collapsed")

    st.markdown('<div class="sidebar-section-label">상품 정보</div>', unsafe_allow_html=True)

    if product_type == "판매 중 상품":
        recommended_items = ["earbuds","headphones","keyboard","mouse","tablet",
                             "speaker","charger","smartwatch","laptop","cable"]
        sel_item  = st.selectbox("추천 검색어", recommended_items)
        search_kw = st.text_input("직접 검색", sel_item)

        search_result = df[df['product_name'].str.contains(search_kw, case=False, na=False)].head(20).reset_index(drop=True)
        options = [f"[{i+1}] {r['product_name'][:55]}" for i, r in search_result.iterrows()]

        if not options:
            st.warning("검색 결과가 없습니다.")
            st.stop()

        sel_opt = st.selectbox("분석 상품 선택", options)
        sel_row = search_result.iloc[options.index(sel_opt)]

        product_name      = sel_row['product_name']
        main_category     = sel_row['main_category']
        display_category  = sel_row['display_category']
        actual_price      = float(sel_row['actual_price'])
        price_level_input = sel_row['price_level']
        about_product     = sel_row.get('about_product', '')

    else:
        product_name  = st.text_input("상품명", "Wireless Headset")
        about_product = st.text_input("설명 키워드", "bluetooth noise cancelling")
        cat_list      = sorted(df['display_category'].dropna().unique())
        display_category = st.selectbox("카테고리", cat_list)
        main_category    = df.loc[df['display_category'] == display_category, 'main_category'].iloc[0]
        actual_price     = st.number_input("예상 판매 가격 (₹)", min_value=0.0, value=5000.0)
        prices = df.loc[df['main_category'] == main_category, 'actual_price']
        q1, q2 = prices.quantile(0.33), prices.quantile(0.66)
        price_level_input = 'Low' if actual_price <= q1 else ('Medium' if actual_price <= q2 else 'High')

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    analyze_button = st.button("분석 시작 →", use_container_width=True)

    # 모델 상태
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex; flex-direction:column; gap:6px;">
        <div class="model-badge">✅ 리뷰 예측 모델 &nbsp; R² = {r2_review}</div>
        <div class="model-badge">✅ 평점 예측 모델 &nbsp; R² = {r2_rating}</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 메인 헤더
# =========================================================
st.markdown("""
<div style="margin-bottom: 4px;">
    <span style="font-size:0.75rem; font-weight:700; text-transform:uppercase;
                 letter-spacing:1.2px; color:#9CA3AF;">Amazon Sales Analytics</span>
</div>
""", unsafe_allow_html=True)
st.title("판매 전략 분석")
st.markdown("""
<p style="font-size:0.95rem; color:#6B7280; margin-top:-8px; margin-bottom:0;">
AI 기반 할인율 시뮬레이션 · 소비자 반응 분석 · 최적 전략 추천
</p>
""", unsafe_allow_html=True)

st.divider()


# =========================================================
# 대기 화면
# =========================================================
if not analyze_button:
    st.markdown("""
    <div class="card" style="padding: 48px 40px; text-align:center;">
        <div style="font-size:2.8rem; margin-bottom:14px;">📊</div>
        <div style="font-size:1.3rem; font-weight:800; color:#1A1D2E; margin-bottom:8px; letter-spacing:-0.4px;">
            왼쪽에서 상품을 선택하세요
        </div>
        <p style="font-size:0.9rem; color:#6B7280; line-height:1.7; max-width:420px; margin:0 auto 28px;">
            판매 중인 상품 또는 신상품 정보를 입력한 뒤<br>
            <b style="color:#4F46E5">분석 시작 →</b> 버튼을 누르면 결과가 나타납니다.
        </p>
        <div class="feature-grid" style="max-width:700px; margin:0 auto;">
            <div class="feature-item">
                <div class="feature-item-icon">🤖</div>
                <div class="feature-item-title">AI 예측 모델</div>
                <div class="feature-item-desc">GradientBoosting 기반 평점·리뷰 수 예측</div>
            </div>
            <div class="feature-item">
                <div class="feature-item-icon">📉</div>
                <div class="feature-item-title">할인율 시뮬레이션</div>
                <div class="feature-item-desc">0~50% 구간별 최적 할인율 자동 계산</div>
            </div>
            <div class="feature-item">
                <div class="feature-item-icon">💬</div>
                <div class="feature-item-title">ABSA 분석</div>
                <div class="feature-item-desc">유사 상품 리뷰 속성별 감성 분석</div>
            </div>
            <div class="feature-item">
                <div class="feature-item-icon">🎯</div>
                <div class="feature-item-title">전략 추천</div>
                <div class="feature-item-desc">시장 데이터 기반 판매 전략 자동 생성</div>
            </div>
            <div class="feature-item">
                <div class="feature-item-icon">🔍</div>
                <div class="feature-item-title">유사 상품 분석</div>
                <div class="feature-item-desc">TF-IDF 기반 경쟁 상품 15개 탐색</div>
            </div>
            <div class="feature-item">
                <div class="feature-item-icon">📍</div>
                <div class="feature-item-title">시장 포지션</div>
                <div class="feature-item-desc">전체 시장 대비 가격·리뷰 위치 시각화</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# =========================================================
# 분석 실행
# =========================================================
with st.spinner("AI 분석 중..."):
    sim_df = simulate_discounts(model_r, model_rt, actual_price, main_category, price_level_input)
    best   = sim_df.sort_values('추천점수', ascending=False).iloc[0]

    user_text = product_name + ' ' + about_product + ' ' + display_category
    scores    = cosine_similarity(vectorizer.transform([user_text]), tfidf_matrix).flatten()
    df['similarity_score'] = scores
    similar   = df[df['product_name'] != product_name].sort_values('similarity_score', ascending=False).head(15).copy()

    absa_raw     = run_absa(similar)
    absa_summary = absa_raw.groupby(['속성','감정'])['언급 횟수'].sum().reset_index() if not absa_raw.empty else absa_raw

    all_text = ' '.join((similar['review_title'].astype(str) + ' ' + similar['review_content'].astype(str)).tolist()).lower()
    kw_df    = pd.DataFrame([[a, sum(all_text.count(k) for k in kws)] for a, kws in ASPECT_KEYWORDS.items()],
                             columns=['속성','언급 횟수']).sort_values('언급 횟수', ascending=False)
    top_aspect = kw_df.iloc[0]['속성']

    discount_strat_name, discount_strat_body = get_discount_strategy(int(best['할인율(%)']))
    aspect_title, aspect_body = ASPECT_COMMENTS.get(top_aspect, (top_aspect, ''))

    price_pct   = (df['actual_price'] < actual_price).mean() * 100
    review_pct  = (df['rating_count'] < best['예상 리뷰 수']).mean() * 100
    rating_pct  = (df['rating'] < best['예상 평점']).mean() * 100
    competition = len(df[df['display_category'] == display_category])


# =========================================================
# 결과 출력
# =========================================================

# ── 분석 대상 카드
st.markdown(f"""
<div class="card">
    <div class="section-label">분석 대상</div>
    <div style="font-size:1.05rem; font-weight:700; color:#1A1D2E; margin-bottom:12px; line-height:1.4;">
        {product_name[:100]}
    </div>
    <span class="tag">{display_category}</span>
    <span class="tag">₹ {int(actual_price):,}</span>
    <span class="tag-gray">가격대 {price_level_input}</span>
    <span class="tag-gray">{product_type}</span>
</div>
""", unsafe_allow_html=True)


# ── 핵심 메트릭
st.markdown('<div class="page-section"><div class="page-section-icon">📊</div><div class="page-section-title">AI 예측 핵심 지표</div></div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("추천 할인율",      f"{int(best['할인율(%)'])}%")
m2.metric("예상 평점",         f"{best['예상 평점']:.2f} / 5.0")
m3.metric("예상 리뷰 수",      f"{best['예상 리뷰 수']:,}개")
m4.metric("시장 경쟁 상품",    f"{competition:,}개")

st.divider()


# ── 할인율 시뮬레이션 차트
st.markdown('<div class="page-section"><div class="page-section-icon">📈</div><div class="page-section-title">할인율별 예측 시뮬레이션</div></div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(line_chart(sim_df['할인율(%)'], sim_df['예상 리뷰 수'],
                               '할인율별 예상 리뷰 수', '할인율(%)', '예상 리뷰 수',
                               best_x=int(best['할인율(%)']), fmt='int'),
                    use_container_width=True)
with c2:
    st.plotly_chart(line_chart(sim_df['할인율(%)'], sim_df['예상 평점'],
                               '할인율별 예상 평점', '할인율(%)', '예상 평점',
                               best_x=int(best['할인율(%)'])),
                    use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(bar_score_chart(sim_df['할인율(%)'], sim_df['추천점수'], int(best['할인율(%)'])),
                    use_container_width=True)
with c4:
    st.plotly_chart(scatter_chart(df, actual_price, np.log1p(best['예상 리뷰 수'])),
                    use_container_width=True)

st.divider()


# ── ABSA
st.markdown('<div class="page-section"><div class="page-section-icon">💬</div><div class="page-section-title">소비자 반응 분석 (ABSA)</div></div>', unsafe_allow_html=True)

ca, cb = st.columns([3, 2])
with ca:
    fig_absa = absa_chart(absa_summary)
    if fig_absa:
        st.plotly_chart(fig_absa, use_container_width=True)
    else:
        st.info("ABSA 분석에 충분한 리뷰 텍스트가 없습니다.")
with cb:
    st.plotly_chart(kw_chart(kw_df), use_container_width=True)

st.divider()


# ── 전략 리포트
st.markdown('<div class="page-section"><div class="page-section-icon">💡</div><div class="page-section-title">AI 판매 전략 리포트</div></div>', unsafe_allow_html=True)

s1, s2 = st.columns(2)
with s1:
    st.markdown(f"""
    <div class="card-highlight">
        <div class="section-label">할인 전략</div>
        <div class="strategy-title">📌 {int(best['할인율(%)'])}% · {discount_strat_name}</div>
        <div class="strategy-body">{discount_strat_body}</div>
    </div>
    """, unsafe_allow_html=True)
with s2:
    st.markdown(f"""
    <div class="card-success">
        <div class="section-label">소비자 소구 전략</div>
        <div class="strategy-title">💬 {aspect_title}</div>
        <div class="strategy-body">{aspect_body}</div>
    </div>
    """, unsafe_allow_html=True)

# 시장 포지션 수치
st.markdown(f"""
<div class="card">
    <div class="section-label">시장 내 위치 분석</div>
    <div style="display:flex; gap:48px; flex-wrap:wrap; margin-top:12px; margin-bottom:16px;">
        <div>
            <div class="big-number-label">가격 상위</div>
            <div class="big-number">{price_pct:.1f}<span style="font-size:1.2rem; font-weight:600; color:#9CA3AF;">%</span></div>
        </div>
        <div>
            <div class="big-number-label">예상 리뷰 상위</div>
            <div class="big-number">{review_pct:.1f}<span style="font-size:1.2rem; font-weight:600; color:#9CA3AF;">%</span></div>
        </div>
        <div>
            <div class="big-number-label">예상 평점 상위</div>
            <div class="big-number">{rating_pct:.1f}<span style="font-size:1.2rem; font-weight:600; color:#9CA3AF;">%</span></div>
        </div>
    </div>
    <div style="font-size:0.88rem; color:#6B7280; line-height:1.7; padding-top:12px; border-top:1px solid #F3F4F6;">
        추천 할인율 <strong style="color:#4F46E5">{int(best['할인율(%)'])}%</strong> 적용 시
        예상 평점 <strong style="color:#1A1D2E">{best['예상 평점']:.2f}점</strong>,
        예상 리뷰 <strong style="color:#1A1D2E">{best['예상 리뷰 수']:,}개</strong>로 예측됩니다.
        (추천점수 산정 기준: 평점 30% + 리뷰 수 30% + 매출 지표 40%)
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# ── 부록
with st.expander("🔍 유사 상품 TOP 15"):
    show = similar[['product_name','display_category','actual_price',
                    'discount_percentage','rating','rating_count','similarity_score']].copy()
    show.columns = ['상품명','카테고리','가격','할인율(%)','평점','리뷰 수','유사도']
    st.dataframe(show.reset_index(drop=True), use_container_width=True)

with st.expander("📋 할인율별 전체 시뮬레이션 테이블"):
    st.dataframe(sim_df[['할인율(%)','할인 후 가격','예상 평점','예상 리뷰 수','추천점수']].reset_index(drop=True),
                 use_container_width=True)
