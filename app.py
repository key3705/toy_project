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
    page_title="SellMate · AI 판매 전략 대시보드",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── 전체 */
.stApp {
    background-color: #F9FAFB;
    font-family: 'Inter', sans-serif;
    color: #111827;
}
.block-container {
    padding-top: 2rem;
    padding-left: 2.6rem;
    padding-right: 2.6rem;
    max-width: 1440px;
}

/* ── 사이드바 */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E5E7EB;
    box-shadow: 1px 0 8px rgba(0,0,0,0.03);
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 0; padding-left: 1.2rem; padding-right: 1.2rem;
}
section[data-testid="stSidebar"] * { color: #111827 !important; }

.sb-logo {
    display: flex; align-items: center; gap: 10px;
    padding: 20px 4px 18px;
    border-bottom: 1px solid #F3F4F6; margin-bottom: 18px;
}
.sb-logo-icon {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 17px;
}
.sb-logo-name {
    font-size: 1.1rem; font-weight: 800; letter-spacing: -0.5px; color: #111827 !important;
}
.sb-logo-name span { color: #4F46E5; }
.sb-logo-sub { font-size: 0.7rem; color: #9CA3AF !important; margin-top: 2px; letter-spacing: 0.2px; }
.sb-label {
    font-size: 0.65rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.3px;
    color: #9CA3AF !important; margin: 16px 0 6px; padding: 0 2px;
}

section[data-testid="stSidebar"] .stRadio > div { gap: 5px; }
section[data-testid="stSidebar"] .stRadio label {
    background: #F9FAFB; border: 1.5px solid #E5E7EB;
    border-radius: 9px; padding: 8px 12px !important;
    font-size: 0.87rem !important; font-weight: 500 !important;
    cursor: pointer; transition: all 0.12s; width: 100%;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: #EEF2FF; border-color: #C7D2FE;
}

section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stNumberInput input {
    background: #F9FAFB !important; border: 1.5px solid #E2E5F1 !important;
    border-radius: 9px !important; color: #111827 !important;
    font-size: 0.87rem !important; padding: 8px 11px !important;
}
section[data-testid="stSidebar"] input:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
    background: #F9FAFB !important; border: 1.5px solid #E2E5F1 !important; border-radius: 9px !important;
}
section[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
    color: #fff !important; border-radius: 10px !important; border: none !important;
    font-size: 0.92rem !important; font-weight: 700 !important;
    padding: 12px 18px !important; width: 100%;
    box-shadow: 0 3px 12px rgba(79,70,229,0.32) !important;
    transition: all 0.18s !important; margin-top: 4px;
}
section[data-testid="stSidebar"] .stButton button:hover {
    box-shadow: 0 5px 18px rgba(79,70,229,0.42) !important;
    transform: translateY(-1px);
}

/* ── 헤더 */
.page-header { padding-bottom: 18px; border-bottom: 1px solid #E5E7EB; margin-bottom: 20px; }
.page-header-eyebrow {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.4px; color: #9CA3AF; margin-bottom: 4px;
}
.page-header-title {
    font-size: 1.6rem; font-weight: 800; color: #111827;
    letter-spacing: -0.6px; line-height: 1.2; margin-bottom: 4px;
}
.page-header-sub { font-size: 0.88rem; color: #6B7280; }

/* ── 상품 요약 카드 */
.product-card {
    background: #fff; border: 1px solid #E5E7EB;
    border-radius: 14px; padding: 18px 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    margin-bottom: 20px;
    display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
}
.product-card-name {
    font-size: 0.98rem; font-weight: 700; color: #111827;
    letter-spacing: -0.2px; flex: 1; min-width: 200px;
}
.product-meta { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }

/* ── 핵심 지표 — 추천 할인율 강조 카드 */
.metric-hero {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    border-radius: 16px; padding: 22px 26px;
    box-shadow: 0 4px 20px rgba(79,70,229,0.28);
    color: #fff;
}
.metric-hero-label {
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.2px; color: rgba(255,255,255,0.7); margin-bottom: 4px;
}
.metric-hero-value {
    font-size: 3rem; font-weight: 800; letter-spacing: -2px; line-height: 1;
    color: #fff; margin-bottom: 4px;
}
.metric-hero-badge {
    display: inline-block; background: rgba(255,255,255,0.18);
    border-radius: 6px; padding: 3px 10px;
    font-size: 0.76rem; font-weight: 600; color: rgba(255,255,255,0.95);
}

.metric-card {
    background: #fff; border: 1px solid #E5E7EB;
    border-radius: 16px; padding: 20px 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    position: relative; overflow: hidden;
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; background: #E5E7EB;
}
.metric-card-label {
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: #9CA3AF; margin-bottom: 6px;
}
.metric-card-value {
    font-size: 1.9rem; font-weight: 800; color: #111827;
    letter-spacing: -1px; line-height: 1;
}

/* 핵심 요약 문장 */
.summary-bar {
    background: #EEF2FF; border: 1px solid #C7D2FE;
    border-radius: 10px; padding: 12px 18px;
    font-size: 0.88rem; color: #3730A3; font-weight: 500;
    line-height: 1.6; margin: 14px 0 0;
}
.summary-bar b { color: #312E81; }

/* ── 섹션 헤더 */
.sec-hdr {
    display: flex; align-items: center; gap: 9px;
    margin: 28px 0 14px;
}
.sec-hdr-icon {
    width: 30px; height: 30px; background: #EEF2FF;
    border-radius: 8px; display: flex; align-items: center;
    justify-content: center; font-size: 14px;
}
.sec-hdr-title { font-size: 1rem; font-weight: 700; color: #111827; letter-spacing: -0.2px; }

/* ── 차트 코멘트 */
.cc { background: #F9FAFB; border: 1px solid #E5E7EB; border-left: 3px solid #4F46E5;
      border-radius: 0 8px 8px 0; padding: 9px 14px;
      font-size: 0.82rem; color: #374151; line-height: 1.6; margin-top: 5px; }
.cc-ok  { background: #F0FDF4; border: 1px solid #BBF7D0; border-left: 3px solid #10B981;
          border-radius: 0 8px 8px 0; padding: 9px 14px;
          font-size: 0.82rem; color: #065F46; line-height: 1.6; margin-top: 5px; }
.cc-warn{ background: #FFFBEB; border: 1px solid #FDE68A; border-left: 3px solid #F59E0B;
          border-radius: 0 8px 8px 0; padding: 9px 14px;
          font-size: 0.82rem; color: #92400E; line-height: 1.6; margin-top: 5px; }

/* ── 전략 카드 */
.strat-card-blue {
    background: #fff; border: 1px solid #E0E7FF; border-left: 4px solid #4F46E5;
    border-radius: 12px; padding: 20px 22px;
    box-shadow: 0 1px 6px rgba(79,70,229,0.07); height: 100%;
}
.strat-card-green {
    background: #fff; border: 1px solid #D1FAE5; border-left: 4px solid #10B981;
    border-radius: 12px; padding: 20px 22px;
    box-shadow: 0 1px 6px rgba(16,185,129,0.07); height: 100%;
}
.strat-label { font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
               letter-spacing: 1.2px; color: #9CA3AF; margin-bottom: 8px; }
.strat-title { font-size: 0.97rem; font-weight: 700; color: #111827;
               margin-bottom: 8px; letter-spacing: -0.2px; }
.strat-body  { font-size: 0.87rem; color: #4B5563; line-height: 1.72; }

/* ── 리뷰 반응 요약 문장 */
.review-summary {
    background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 10px;
    padding: 14px 18px; margin-top: 10px;
    font-size: 0.87rem; color: #374151; line-height: 1.65;
}
.review-summary b { color: #111827; }

/* ── 태그 */
.tag  { display:inline-flex; align-items:center; background:#EEF2FF; color:#4F46E5;
        border-radius:5px; padding:3px 10px; font-size:0.75rem; font-weight:600;
        margin-right:5px; margin-bottom:3px; }
.tag-g{ display:inline-flex; align-items:center; background:#F3F4F6; color:#6B7280;
        border-radius:5px; padding:3px 10px; font-size:0.75rem; font-weight:600;
        margin-right:5px; margin-bottom:3px; }

/* ── 시장 포지션 숫자 */
.pos-num { font-size: 2.4rem; font-weight: 800; color: #4F46E5;
           letter-spacing: -1.5px; line-height: 1; }
.pos-lbl { font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
           letter-spacing: 0.8px; color: #9CA3AF; margin-bottom: 4px; }

/* ── 빈도/ABSA 인라인 테이블 */
.ftable { width:100%; border-collapse:collapse; font-size:0.84rem; }
.ftable th { background:#F9FAFB; color:#6B7280; font-weight:700; font-size:0.67rem;
             text-transform:uppercase; letter-spacing:0.6px;
             padding:7px 10px; text-align:left; border-bottom:1px solid #E5E7EB; }
.ftable td { padding:8px 10px; border-bottom:1px solid #F3F4F6;
             color:#111827; vertical-align:middle; }
.ftable tr:last-child td { border-bottom:none; }
.fbar-bg   { background:#EEF2FF; border-radius:3px; height:7px; width:100%; }
.fbar-fill { background:#4F46E5; border-radius:3px; height:7px; }
.sp { display:inline-block; border-radius:999px; padding:2px 9px;
      font-size:0.72rem; font-weight:700; margin:1px; }

/* ── 모델 배지 */
.mbadge { display:inline-flex; align-items:center; gap:5px;
          background:#F0FDF4; border:1px solid #BBF7D0; border-radius:7px;
          padding:4px 10px; font-size:0.75rem; font-weight:600; color:#065F46; }

section[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
    color: #fff !important; border-radius: 10px !important; border: none !important;
    font-size: 0.92rem !important; font-weight: 700 !important;
    padding: 12px 18px !important; width: 100%;
    box-shadow: 0 3px 12px rgba(79,70,229,0.32) !important;
    transition: all 0.18s !important; margin-top: 4px;
}
section[data-testid="stSidebar"] .stButton button:hover {
    box-shadow: 0 5px 18px rgba(79,70,229,0.42) !important;
    transform: translateY(-1px);
}



/* ── misc */
[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; border:1px solid #E5E7EB !important; }
.stDivider { border-color:#F3F4F6 !important; }
h1,h2,h3 { color:#111827 !important; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 데이터 + 모델
# =========================================================
@st.cache_data(show_spinner=False)
def load_and_preprocess():
    df = pd.read_csv("amazon.csv")

    def clean_col(s, chars):
        r = s.astype(str)
        for c in chars: r = r.str.replace(c, '', regex=False)
        return r

    df['actual_price']        = pd.to_numeric(clean_col(df['actual_price'],        ['₹',',']), errors='coerce')
    df['discounted_price']    = pd.to_numeric(clean_col(df['discounted_price'],    ['₹',',']), errors='coerce')
    df['discount_percentage'] = pd.to_numeric(clean_col(df['discount_percentage'], ['%']),     errors='coerce')
    df['rating']              = pd.to_numeric(df['rating'],                                    errors='coerce')
    df['rating_count']        = pd.to_numeric(clean_col(df['rating_count'],        [',']),     errors='coerce')

    df['main_category'] = df['category'].astype(str).str.split('|').str[0]
    cat_map = {
        "Computers&Accessories":"💻 Computer","Electronics":"🎧 Electronics",
        "Home&Kitchen":"🏠 Home","OfficeProducts":"🖨 Office",
        "MusicalInstruments":"🎵 Music","Health&PersonalCare":"💖 Health",
        "HomeImprovement":"🛠 Tools","Toys&Games":"🧸 Toys"
    }
    df['display_category'] = df['main_category'].map(cat_map).fillna("✨ Other")

    df = df.dropna(subset=['product_name','actual_price','discount_percentage',
                            'rating','rating_count','display_category',
                            'about_product','review_title','review_content']).reset_index(drop=True)

    def price_level(row):
        p = df.loc[df['main_category']==row['main_category'],'actual_price']
        q1,q2 = p.quantile(0.33), p.quantile(0.66)
        return 'Low' if row['actual_price']<=q1 else ('Medium' if row['actual_price']<=q2 else 'High')

    df['price_level']                  = df.apply(price_level, axis=1)
    df['log_rating_count']             = np.log1p(df['rating_count'])
    df['price_discount_interaction']   = df['actual_price'] * df['discount_percentage']
    df['consumer_score']               = df['rating'] * np.log1p(df['rating_count'])
    df['combined_text'] = (df['product_name'].fillna('')+' '+
                           df['about_product'].fillna('')+' '+
                           df['display_category'].fillna(''))
    return df


@st.cache_resource(show_spinner=False)
def train_models(df):
    np_ = Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())])
    cp_ = Pipeline([('i',SimpleImputer(strategy='most_frequent')),('o',OneHotEncoder(handle_unknown='ignore'))])

    def fit(feat,num,cat,y_col):
        pre = ColumnTransformer([('n',np_,num),('c',cp_,cat)])
        mdl = Pipeline([('p',pre),('r',GradientBoostingRegressor(n_estimators=100,random_state=42))])
        X,y = df[feat], df[y_col]
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        mdl.fit(Xtr,ytr)
        return mdl, round(r2_score(yte,mdl.predict(Xte)),3)

    mr,r2r = fit(
        ['actual_price','discount_percentage','price_discount_interaction','main_category','price_level','rating'],
        ['actual_price','discount_percentage','price_discount_interaction','rating'],
        ['main_category','price_level'], 'log_rating_count')
    mrt,r2rt = fit(
        ['actual_price','discount_percentage','price_discount_interaction','main_category','price_level'],
        ['actual_price','discount_percentage','price_discount_interaction'],
        ['main_category','price_level'], 'rating')
    return mr, mrt, r2r, r2rt


@st.cache_resource(show_spinner=False)
def build_tfidf(df):
    v = TfidfVectorizer(stop_words='english', max_features=8000)
    m = v.fit_transform(df['combined_text'])
    return v, m


# =========================================================
# ABSA
# =========================================================
AK = {
    '가격/가성비':['price','cheap','expensive','value','worth','budget','cost'],
    '품질':       ['quality','durable','material','build','sturdy'],
    '배송/포장':  ['delivery','shipping','package','packing','arrived'],
    '사용성':     ['easy','comfortable','convenient','install','simple'],
    '디자인':     ['design','look','style','color','appearance'],
    '성능':       ['performance','fast','battery','sound','noise','power','speed']
}
PW = ['good','great','excellent','amazing','fast','love','perfect','easy',
      'comfortable','nice','best','awesome','satisfied','worth','happy']
NW = ['bad','poor','terrible','worst','slow','hate','broken','problem',
      'difficult','hard','noise','damaged','waste','issue','disappoint']

def run_absa(pdf):
    rows=[]
    for _,row in pdf.iterrows():
        txt=(str(row['review_title'])+' '+str(row['review_content'])).lower().split()
        for asp,kws in AK.items():
            pos=neg=0
            for kw in kws:
                for i,w in enumerate(txt):
                    if kw in w:
                        win=txt[max(0,i-3):min(len(txt),i+4)]
                        for nw in win:
                            if nw in PW: pos+=1
                            elif nw in NW: neg+=1
            tot=pos+neg
            if tot>0:
                sent='긍정' if pos>neg else ('부정' if neg>pos else '중립')
                rows.append([asp,sent,tot,pos,neg])
    if not rows: return pd.DataFrame(columns=['속성','감정','언급 횟수','긍정','부정'])
    return pd.DataFrame(rows,columns=['속성','감정','언급 횟수','긍정','부정'])


# =========================================================
# 시뮬레이션
# =========================================================
def simulate(mr, mrt, price, mc, pl):
    res=[]
    for d in range(0,51,5):
        ix=price*d
        ri=pd.DataFrame({'actual_price':[price],'discount_percentage':[d],
                         'price_discount_interaction':[ix],'main_category':[mc],'price_level':[pl]})
        rt=float(np.clip(mrt.predict(ri)[0],1,5))
        rvi=pd.DataFrame({'actual_price':[price],'discount_percentage':[d],
                          'price_discount_interaction':[ix],'main_category':[mc],
                          'price_level':[pl],'rating':[rt]})
        rv=max(0,int(np.expm1(mr.predict(rvi)[0])))
        res.append([d,int(price*(1-d/100)),round(rt,1),rv])
    s=pd.DataFrame(res,columns=['할인율(%)','할인 후 가격','예상 평점','예상 리뷰 수'])
    s['리뷰수_N'] = s['예상 리뷰 수']/(s['예상 리뷰 수'].max()+1e-9)
    s['평점_N']   = s['예상 평점']/5
    s['매출지표'] = s['할인 후 가격']*s['예상 리뷰 수']
    s['매출_N']   = s['매출지표']/(s['매출지표'].max()+1e-9)
    s['추천점수'] = ((s['평점_N']*0.3+s['리뷰수_N']*0.3+s['매출_N']*0.4)*100).round(1)
    return s


# =========================================================
# 전략 텍스트
# =========================================================
AC = {
    '가격/가성비':('가성비 강조 전략','유사 상품 리뷰에서 가격 대비 만족도 언급이 가장 많았습니다. 상세페이지에서 가성비와 할인 혜택을 전면에 내세우는 것이 효과적입니다.'),
    '품질':       ('품질 신뢰성 전략','소비자들이 내구성·완성도를 중점적으로 언급했습니다. 인증 자료나 소재 정보를 상세페이지 상단에 배치하세요.'),
    '배송/포장':  ('물류 경쟁력 전략','배송 속도·포장 상태에 대한 반응이 높았습니다. 빠른 배송과 안전 포장을 마케팅 핵심 메시지로 활용하세요.'),
    '사용성':     ('편의성 소구 전략','사용 편의성과 직관적 조작법에 대한 언급이 많았습니다. 사용 가이드 영상이나 간편 설치 포인트를 강조하세요.'),
    '디자인':     ('비주얼 마케팅 전략','디자인·외형 관련 피드백이 높게 나타났습니다. 감성적인 제품 연출 이미지와 색상 다양성을 전면에 배치하세요.'),
    '성능':       ('스펙 중심 전략','성능 관련 언급이 가장 강했습니다. 배터리, 속도, 출력 등 핵심 스펙을 수치와 함께 소구하는 전략이 유리합니다.')
}

def discount_strat(d):
    if d<=10:  return '수익성 유지형','낮은 할인율에서도 평점·리뷰가 안정적으로 유지됩니다. 과도한 할인보다 마진 방어에 집중하는 것이 유리합니다.'
    elif d<=20: return '균형형','중간 할인율에서 리뷰 수·평점·매출의 밸런스가 최적화됩니다. 가격 경쟁력과 마진을 동시에 확보하는 전략입니다.'
    else:       return '공격형 프로모션','높은 할인율에서 리뷰 유입 효과가 크게 나타납니다. 단기 노출 확대와 리뷰 대량 확보를 목표로 한 전략에 적합합니다.'


# =========================================================
# 핵심 요약 문장
# =========================================================
def make_summary(best, sim):
    d  = int(best['할인율(%)'])
    rv = int(best['예상 리뷰 수'])
    rt = best['예상 평점']
    sc = best['추천점수']
    rv0 = int(sim.loc[sim['할인율(%)']==0,'예상 리뷰 수'].values[0])
    delta = rv - rv0
    delta_txt = f"할인 없을 때보다 리뷰 <b>{delta:+,}개</b> 차이," if delta != 0 else ""
    return (f"현재 상품은 <b>{d}% 할인</b> 시 예상 리뷰 수·평점·매출 지표가 가장 균형적입니다. "
            f"{delta_txt} 예상 평점 <b>{rt:.1f}점</b>으로 추천점수 <b>{sc:.1f}점</b>을 기록합니다.")


# =========================================================
# 리뷰 반응 요약 문장
# =========================================================
def make_review_summary(top_aspect, kw_df):
    top2 = kw_df.head(2)['속성'].tolist()
    t2_str = "·".join(f"'{a}'" for a in top2)
    detail = {
        '가격/가성비': '상세페이지에서 가격 경쟁력과 할인 혜택을 함께 강조하는 것이 좋습니다.',
        '품질':        '내구성·소재 품질을 증명하는 인증 자료나 사용자 후기를 전면에 내세우세요.',
        '배송/포장':   '빠른 배송 일정과 안전 포장 정보를 상세페이지 상단에 배치하세요.',
        '사용성':      '간편한 설치 방법이나 사용 튜토리얼 콘텐츠를 제공하면 전환율이 높아집니다.',
        '디자인':      '고품질 제품 연출 이미지와 색상·디자인 다양성을 강조하세요.',
        '성능':        '배터리 타임, 처리 속도 등 핵심 스펙을 수치와 함께 비교 표로 보여주세요.'
    }
    return (f"이 상품군에서는 {t2_str} 관련 언급이 가장 많습니다. "
            f"{detail.get(top_aspect,'관련 강점을 상세페이지에 적극 활용하세요.')}")


# =========================================================
# 차트 코멘트
# =========================================================
def cc_review(sim, best):
    d  = int(best['할인율(%)'])
    rv = int(best['예상 리뷰 수'])
    rv0= int(sim.loc[sim['할인율(%)']==0,'예상 리뷰 수'].values[0])
    delta = rv - rv0
    if d==0:
        return 'cc','할인 없이도 리뷰 유입이 안정적입니다. 불필요한 할인을 줄이고 상세페이지 품질에 집중하세요.'
    elif delta>0:
        return 'cc-ok',f'<b>{d}% 할인</b> 시 리뷰가 할인 없을 때보다 약 <b>{delta:,}개</b> 더 많이 예측됩니다. 이 구간까지 할인이 소비자 유입에 효과적입니다.'
    else:
        return 'cc-warn','할인율을 높여도 리뷰 수 증가 효과가 작습니다. 노출 전략이나 상세페이지 개선을 병행하세요.'

def cc_rating(sim, best):
    d  = int(best['할인율(%)'])
    rt = best['예상 평점']
    diff = round(sim['예상 평점'].max()-sim['예상 평점'].min(),1)
    if diff<0.1:
        return 'cc','할인율 변화에 따른 평점 차이가 미미합니다. 평점은 상품 자체의 품질에 더 크게 좌우됩니다.'
    elif rt>=4.0:
        return 'cc-ok',f'<b>{d}% 할인</b> 구간에서 예상 평점 <b>{rt:.1f}점</b>으로 소비자 만족도가 최적화됩니다.'
    else:
        return 'cc-warn',f'최고 예상 평점이 {rt:.1f}점으로 낮은 편입니다. 상품 품질 개선이나 상세페이지 보완을 권장합니다.'

def cc_score(best):
    d=int(best['할인율(%)']); s=best['추천점수']
    if d<=10:   return 'cc',f'<b>{d}%</b>의 낮은 할인율에서 추천점수 <b>{s:.1f}점</b>으로 최고입니다. 마진을 지키면서도 충분한 소비자 반응을 기대할 수 있습니다.'
    elif d<=25: return 'cc-ok',f'<b>{d}% 할인</b>이 추천점수 <b>{s:.1f}점</b>으로 평점·리뷰·매출의 균형이 가장 좋습니다.'
    else:       return 'cc-warn',f'<b>{d}%</b>의 높은 할인율이 최적입니다. 단기 리뷰 확보 목적에는 유효하지만, 마진 훼손 가능성을 반드시 검토하세요.'

def cc_scatter(pp, rp):
    if pp>70 and rp>50: return 'cc-ok', f'가격 상위 {pp:.0f}% 구간이지만 예상 리뷰도 상위 {rp:.0f}% 수준입니다. 프리미엄 포지셔닝을 유지하면서 리뷰 관리에 집중하세요.'
    elif pp>70:         return 'cc-warn',f'가격이 시장 상위 {pp:.0f}%로 높은 편입니다. 프리미엄 이미지 강화 또는 소비자 체감 가치를 높이는 전략이 필요합니다.'
    else:               return 'cc',    f'가격 경쟁력이 있는 구간(하위 {100-pp:.0f}%)에 위치합니다. 가성비를 전면에 내세우는 마케팅이 효과적입니다.'


# =========================================================
# 차트
# =========================================================
C={'i':'#4F46E5','v':'#7C3AED','r':'#EF4444','g':'#10B981',
   'gy':'#E5E7EB','t':'#111827','st':'#6B7280','gr':'#F3F4F6','bg':'#FFFFFF'}
MB=dict(displaylogo=False,scrollZoom=True,
        modeBarButtonsToRemove=['zoom2d','select2d','lasso2d','zoomIn2d','zoomOut2d',
                                'autoScale2d','hoverClosestCartesian','hoverCompareCartesian',
                                'toggleSpikelines','toImage'],
        modeBarButtonsToAdd=['pan2d','resetScale2d'], displayModeBar=True)
CL=dict(paper_bgcolor=C['bg'],plot_bgcolor=C['bg'],
        font=dict(family='Inter, sans-serif',color=C['t']),
        margin=dict(l=14,r=14,t=48,b=28),
        hoverlabel=dict(bgcolor='white',font_size=13,font_family='Inter'))

def lc(x,y,title,xl,yl,bx=None,fmt='int'):
    ya=list(y); ym=max(ya) if ya else 1
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=x,y=y,fill='tozeroy',fillcolor='rgba(79,70,229,0.07)',
                              line=dict(color=C['i'],width=0),showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x,y=y,mode='lines+markers+text',
                              line=dict(color=C['i'],width=2.5),
                              marker=dict(size=8,color='white',line=dict(color=C['i'],width=2.5)),
                              text=[f"{v:,}" if fmt=='int' else f"{v:.1f}" for v in ya],
                              textposition='top center',textfont=dict(size=10,color=C['st']),
                              hovertemplate=f'<b>%{{x}}%</b><br>{yl}: %{{y}}<extra></extra>',name=yl))
    if bx is not None:
        fig.add_vline(x=bx,line_dash='dot',line_color=C['r'],line_width=1.8,
                      annotation_text=f"  추천 {bx}%",annotation_position="top right",
                      annotation_font=dict(color=C['r'],size=12,family='Inter'))
    fig.update_layout(title=dict(text=title,font=dict(size=13,color=C['t']),x=0,xanchor='left'),
                      xaxis=dict(title=xl,gridcolor=C['gr'],showgrid=True,zeroline=False,tickfont=dict(size=10)),
                      yaxis=dict(title=yl,gridcolor=C['gr'],showgrid=True,zeroline=False,
                                 tickfont=dict(size=10),range=[0,ym*1.22]),
                      showlegend=False,height=290,**CL)
    return fig

def bc(x,y,bx):
    colors=[C['i'] if xi==bx else C['gy'] for xi in x]
    ym=max(y) if len(y) else 100
    fig=go.Figure(go.Bar(x=[str(v) for v in x],y=y,
                          marker=dict(color=colors,line=dict(width=0),cornerradius=5),
                          text=[f"{v:.1f}점" for v in y],textposition='outside',
                          textfont=dict(size=10,color=C['st']),
                          hovertemplate='할인율 %{x}%<br>추천점수: %{y:.1f}점<extra></extra>'))
    fig.update_layout(title=dict(text='할인율별 종합 추천점수 (100점 만점)',font=dict(size=13,color=C['t']),x=0,xanchor='left'),
                      xaxis=dict(title='할인율(%)',gridcolor=C['gr'],zeroline=False,tickfont=dict(size=10)),
                      yaxis=dict(title='추천점수 (점)',gridcolor=C['gr'],showgrid=True,zeroline=False,
                                 tickfont=dict(size=10),range=[0,ym*1.2]),
                      height=290,**CL)
    return fig

def ac(adf):
    if adf.empty: return None
    pv=adf.groupby(['속성','감정'])['언급 횟수'].sum().reset_index()
    cm={'긍정':C['i'],'중립':'#A5B4FC','부정':C['r']}
    fig=px.bar(pv,x='속성',y='언급 횟수',color='감정',barmode='stack',
               color_discrete_map=cm,text='언급 횟수')
    fig.update_traces(marker_line_width=0,textfont_size=10,textposition='inside',insidetextanchor='middle')
    fig.update_layout(title=dict(text='속성별 소비자 반응',font=dict(size=13,color=C['t']),x=0,xanchor='left'),
                      xaxis=dict(gridcolor=C['gr'],zeroline=False,tickfont=dict(size=10)),
                      yaxis=dict(gridcolor=C['gr'],showgrid=True,zeroline=False,tickfont=dict(size=10)),
                      legend=dict(title='',font=dict(size=10),orientation='h',y=1.14),
                      height=290,**CL)
    return fig

def kc(kdf):
    ds=kdf.sort_values('언급 횟수')
    xm=ds['언급 횟수'].max() if len(ds) else 10
    fig=go.Figure(go.Bar(x=ds['언급 횟수'],y=ds['속성'],orientation='h',
                          marker=dict(color=ds['언급 횟수'],colorscale=[[0,'#E0E7FF'],[1,C['i']]],line=dict(width=0)),
                          text=[f"  {v}회" for v in ds['언급 횟수']],textposition='outside',
                          textfont=dict(size=10,color=C['st']),
                          hovertemplate='%{y}: %{x}회<extra></extra>'))
    fig.update_layout(title=dict(text='속성별 언급 빈도',font=dict(size=13,color=C['t']),x=0,xanchor='left'),
                      xaxis=dict(gridcolor=C['gr'],showgrid=True,zeroline=False,
                                 tickfont=dict(size=10),range=[0,xm*1.28]),
                      yaxis=dict(gridcolor=C['gr'],zeroline=False,tickfont=dict(size=10)),
                      height=290,**CL)
    return fig

def sc(df,ip,plrv):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df['actual_price'],y=df['log_rating_count'],mode='markers',
                              marker=dict(size=5,color=df['rating'],colorscale='Blues',opacity=0.4,
                                         showscale=True,colorbar=dict(title='평점',thickness=9,len=0.75)),
                              hovertemplate='가격: ₹%{x:,}<br>log(리뷰): %{y:.2f}<extra></extra>',name='전체 상품'))
    fig.add_trace(go.Scatter(x=[ip],y=[plrv],mode='markers',
                              marker=dict(size=18,symbol='star',color=C['r'],line=dict(color='white',width=2)),
                              name='선택 상품',
                              hovertemplate='선택 상품<br>가격: ₹%{x:,}<br>log(리뷰): %{y:.2f}<extra></extra>'))
    fig.update_layout(title=dict(text='시장 내 상품 포지션',font=dict(size=13,color=C['t']),x=0,xanchor='left'),
                      xaxis=dict(title='가격 (₹)',gridcolor=C['gr'],showgrid=True,zeroline=False),
                      yaxis=dict(title='log(리뷰 수+1)',gridcolor=C['gr'],showgrid=True,zeroline=False),
                      legend=dict(font=dict(size=10),orientation='h',y=1.14),
                      height=290,**CL)
    return fig


# =========================================================
# 인라인 수치 테이블
# =========================================================
def kw_table_html(kdf):
    mx=kdf['언급 횟수'].max() if len(kdf) else 1
    rows=""
    for _,r in kdf.iterrows():
        p=int(r['언급 횟수']/mx*100)
        rows+=f"""<tr>
          <td style="font-weight:600;">{r['속성']}</td>
          <td style="width:110px;"><div class="fbar-bg"><div class="fbar-fill" style="width:{p}%;"></div></div></td>
          <td style="text-align:right;font-weight:700;color:#4F46E5;">{int(r['언급 횟수'])}회</td></tr>"""
    return (f'<table class="ftable"><thead><tr><th>속성</th><th>비율</th><th>횟수</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>')

def absa_table_html(asum):
    if asum.empty: return '<p style="color:#9CA3AF;font-size:0.82rem;">분석 결과 없음</p>'
    pv=asum.pivot_table(index='속성',columns='감정',values='언급 횟수',aggfunc='sum',fill_value=0).reset_index()
    for c in ['긍정','중립','부정']:
        if c not in pv.columns: pv[c]=0
    sty={'긍정':'background:#EEF2FF;color:#4F46E5;','중립':'background:#F3F4F6;color:#6B7280;',
         '부정':'background:#FEF2F2;color:#EF4444;'}
    rows=""
    for _,r in pv.iterrows():
        pills="".join(f'<span class="sp" style="{sty[s]}">{s} {int(r.get(s,0))}</span>'
                      for s in ['긍정','중립','부정'] if int(r.get(s,0))>0)
        rows+=f'<tr><td style="font-weight:600;">{r["속성"]}</td><td>{pills}</td></tr>'
    return (f'<table class="ftable"><thead><tr><th>속성</th><th>긍정 / 중립 / 부정 (언급 수)</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>')


# =========================================================
# 초기화
# =========================================================
with st.spinner("데이터 로딩 중..."):
    df = load_and_preprocess()
with st.spinner("AI 모델 준비 중 (최초 1회)..."):
    mr, mrt, r2r, r2rt = train_models(df)
    vec, tmat = build_tfidf(df)


# =========================================================
# 사이드바
# =========================================================
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
      <div class="sb-logo-icon">🤝</div>
      <div>
        <div class="sb-logo-name">Sell<span>Mate</span></div>
        <div class="sb-logo-sub">판매자의 AI 파트너</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-label">STEP 1 · 상품 유형 선택</div>', unsafe_allow_html=True)
    ptype = st.radio("", ["판매 중 상품 선택", "신상품 직접 입력"], label_visibility="collapsed")

    st.markdown('<div class="sb-label">STEP 2 · 상품 정보 입력</div>', unsafe_allow_html=True)

    if ptype == "판매 중 상품 선택":
        rec = ["earbuds","headphones","keyboard","mouse","tablet",
               "speaker","charger","smartwatch","laptop","cable"]

        # 추천 검색어 selectbox (빠른 선택)
        sel_item = st.selectbox(
            "추천 검색어",
            rec,
            help="자주 찾는 검색어를 선택하면 아래 검색창에 자동 입력됩니다."
        )

        # 직접 검색 (자유 입력) — selectbox 선택값이 기본값
        skw = st.text_input(
            "🔍 직접 검색",
            value=sel_item,
            placeholder="상품명을 영어로 입력하세요",
            help="직접 입력하면 더 정확한 검색이 가능합니다."
        )

        sv = vec.transform([skw])
        ss = cosine_similarity(sv, tmat).flatten()
        df['_ss'] = ss
        nm = df[df['product_name'].str.contains(skw, case=False, na=False)]
        sr = (nm.sort_values('_ss', ascending=False).head(20) if len(nm)>=3
              else df.sort_values('_ss', ascending=False).head(20)).reset_index(drop=True)

        opts = [f"[{i+1}] {r['product_name'][:52]}" for i,r in sr.iterrows()]
        if not opts:
            st.warning("검색 결과가 없습니다."); st.stop()

        sel = st.selectbox("분석 상품 선택", opts)
        row = sr.iloc[opts.index(sel)]
        pname  = row['product_name']
        mcat   = row['main_category']
        dcat   = row['display_category']
        price  = float(row['actual_price'])
        pl     = row['price_level']
        about  = str(row.get('about_product',''))

    else:
        pname = st.text_input("상품명", "Wireless Headset")
        about = st.text_input("설명 키워드", "bluetooth noise cancelling")
        cats  = sorted(df['display_category'].dropna().unique())
        dcat  = st.selectbox("카테고리", cats)
        mcat  = df.loc[df['display_category']==dcat,'main_category'].iloc[0]
        price = st.number_input("예상 판매 가격 (₹)", min_value=0.0, value=5000.0)
        ps    = df.loc[df['main_category']==mcat,'actual_price']
        q1,q2 = ps.quantile(0.33), ps.quantile(0.66)
        pl    = 'Low' if price<=q1 else ('Medium' if price<=q2 else 'High')

    st.markdown('<div class="sb-label">STEP 3 · 분석 실행</div>', unsafe_allow_html=True)
    go_btn = st.button("분석 시작 →", use_container_width=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;gap:5px;">
      <div class="mbadge">✅ 리뷰 예측 모델 R² = {r2r}</div>
      <div class="mbadge">✅ 평점 예측 모델 R² = {r2rt}</div>
    </div>""", unsafe_allow_html=True)


# =========================================================
# 메인 헤더
# =========================================================
st.markdown("""
<div class="page-header">
  <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
    <div style="font-size:1.55rem; font-weight:900; color:#111827; letter-spacing:-1px; line-height:1;">
      Sell<span style="color:#4F46E5;">Mate</span>
    </div>
    <div style="background:#EEF2FF; color:#4F46E5; border-radius:6px; padding:3px 10px;
                font-size:0.72rem; font-weight:700; letter-spacing:0.5px;">
      AI 판매 전략 대시보드
    </div>
  </div>
  <div class="page-header-sub">
    판매자의 AI 파트너 · 상품 데이터를 기반으로 <b style="color:#111827;">최적 할인율</b>과
    <b style="color:#111827;">소비자 반응 전략</b>을 자동으로 추천합니다.
  </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 대기 화면
# =========================================================
if not go_btn:
    st.markdown("""
    <div style="background:#fff;border:1px solid #E5E7EB;border-radius:14px;
                padding:52px 40px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
      <div style="font-size:2.8rem;font-weight:900;color:#111827;letter-spacing:-2px;margin-bottom:6px;">
        Sell<span style="color:#4F46E5;">Mate</span>
      </div>
      <div style="font-size:0.82rem;color:#9CA3AF;letter-spacing:0.5px;margin-bottom:20px;">
        판매자의 AI 파트너
      </div>
      <div style="width:48px;height:2px;background:linear-gradient(90deg,#4F46E5,#7C3AED);
                  border-radius:2px;margin:0 auto 24px;"></div>
      <div style="font-size:1.1rem;font-weight:700;color:#111827;margin-bottom:8px;letter-spacing:-0.3px;">
        왼쪽 사이드바에서 상품을 선택하세요
      </div>
      <p style="font-size:0.88rem;color:#6B7280;line-height:1.7;max-width:400px;margin:0 auto 28px;">
        판매 중인 상품 또는 신상품 정보를 입력한 뒤<br>
        <b style="color:#4F46E5">분석 시작 →</b> 버튼을 누르면 AI 분석 결과가 나타납니다.
      </p>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;max-width:680px;margin:0 auto;">
        <div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:10px;padding:18px;text-align:left;">
          <div style="font-size:1.4rem;margin-bottom:8px;">🤖</div>
          <div style="font-size:0.87rem;font-weight:700;color:#111827;margin-bottom:3px;">AI 예측 모델</div>
          <div style="font-size:0.78rem;color:#6B7280;line-height:1.5;">GradientBoosting 기반 평점·리뷰 수 예측</div>
        </div>
        <div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:10px;padding:18px;text-align:left;">
          <div style="font-size:1.4rem;margin-bottom:8px;">📉</div>
          <div style="font-size:0.87rem;font-weight:700;color:#111827;margin-bottom:3px;">할인율 시뮬레이션</div>
          <div style="font-size:0.78rem;color:#6B7280;line-height:1.5;">0~50% 구간별 최적 할인율 자동 계산</div>
        </div>
        <div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:10px;padding:18px;text-align:left;">
          <div style="font-size:1.4rem;margin-bottom:8px;">💬</div>
          <div style="font-size:0.87rem;font-weight:700;color:#111827;margin-bottom:3px;">리뷰 반응 분석</div>
          <div style="font-size:0.78rem;color:#6B7280;line-height:1.5;">유사 상품 리뷰 속성별 감성 분석</div>
        </div>
        <div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:10px;padding:18px;text-align:left;">
          <div style="font-size:1.4rem;margin-bottom:8px;">🎯</div>
          <div style="font-size:0.87rem;font-weight:700;color:#111827;margin-bottom:3px;">판매 전략 추천</div>
          <div style="font-size:0.78rem;color:#6B7280;line-height:1.5;">시장 데이터 기반 최적 전략 자동 생성</div>
        </div>
        <div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:10px;padding:18px;text-align:left;">
          <div style="font-size:1.4rem;margin-bottom:8px;">🔍</div>
          <div style="font-size:0.87rem;font-weight:700;color:#111827;margin-bottom:3px;">유사 상품 분석</div>
          <div style="font-size:0.78rem;color:#6B7280;line-height:1.5;">TF-IDF 기반 경쟁 상품 15개 탐색</div>
        </div>
        <div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:10px;padding:18px;text-align:left;">
          <div style="font-size:1.4rem;margin-bottom:8px;">📍</div>
          <div style="font-size:0.87rem;font-weight:700;color:#111827;margin-bottom:3px;">시장 포지션</div>
          <div style="font-size:0.78rem;color:#6B7280;line-height:1.5;">전체 시장 대비 가격·리뷰 위치 시각화</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# =========================================================
# 분석
# =========================================================
with st.spinner("AI 분석 중..."):
    sim  = simulate(mr, mrt, price, mcat, pl)
    best = sim.sort_values('추천점수', ascending=False).iloc[0]

    utxt  = pname+' '+about+' '+dcat
    scos  = cosine_similarity(vec.transform([utxt]), tmat).flatten()
    df['similarity_score'] = scos
    sim15 = df[df['product_name']!=pname].sort_values('similarity_score', ascending=False).head(15).copy()

    absa_raw  = run_absa(sim15)
    absa_sum  = (absa_raw.groupby(['속성','감정'])['언급 횟수'].sum().reset_index()
                 if not absa_raw.empty else absa_raw)

    at   = ' '.join((sim15['review_title'].astype(str)+' '+sim15['review_content'].astype(str)).tolist()).lower()
    kwdf = pd.DataFrame([[a,sum(at.count(k) for k in kws)] for a,kws in AK.items()],
                         columns=['속성','언급 횟수']).sort_values('언급 횟수', ascending=False)
    top_asp = kwdf.iloc[0]['속성']

    dsn, dsb = discount_strat(int(best['할인율(%)']))
    atn, atb = AC.get(top_asp, (top_asp,''))
    pp  = (df['actual_price'] < price).mean()*100
    rp  = (df['rating_count'] < best['예상 리뷰 수']).mean()*100
    rtp = (df['rating'] < best['예상 평점']).mean()*100
    comp= len(df[df['display_category']==dcat])


# =========================================================
# ① 선택 상품 요약 카드
# =========================================================
st.markdown(f"""
<div class="product-card">
  <div class="product-card-name">{pname[:100]}</div>
  <div class="product-meta">
    <span class="tag">{dcat}</span>
    <span class="tag">₹ {int(price):,}</span>
    <span class="tag-g">가격대 {pl}</span>
    <span class="tag-g">{ptype}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# ② 핵심 지표 — 추천 할인율 강조 + 3개 일반 카드
# =========================================================
st.markdown('<div class="sec-hdr"><div class="sec-hdr-icon">📊</div><div class="sec-hdr-title">핵심 결과</div></div>', unsafe_allow_html=True)

c_hero, c2, c3, c4 = st.columns([1.4, 1, 1, 1])

with c_hero:
    st.markdown(f"""
    <div class="metric-hero">
      <div class="metric-hero-label">추천 할인율</div>
      <div class="metric-hero-value">{int(best['할인율(%)'])}%</div>
      <div class="metric-hero-badge">{dsn}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-card-label">예상 평점</div>
      <div class="metric-card-value">{best['예상 평점']:.1f}<span style="font-size:1rem;font-weight:500;color:#9CA3AF;"> / 5.0</span></div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-card-label">예상 리뷰 수</div>
      <div class="metric-card-value">{int(best['예상 리뷰 수']):,}<span style="font-size:1rem;font-weight:500;color:#9CA3AF;">개</span></div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-card-label">시장 경쟁 상품</div>
      <div class="metric-card-value">{comp:,}<span style="font-size:1rem;font-weight:500;color:#9CA3AF;">개</span></div>
    </div>
    """, unsafe_allow_html=True)

# 핵심 요약 문장
st.markdown(f'<div class="summary-bar">{make_summary(best, sim)}</div>', unsafe_allow_html=True)

st.divider()


# =========================================================
# ③ 할인율 시뮬레이션
# =========================================================
st.markdown('<div class="sec-hdr"><div class="sec-hdr-icon">📈</div><div class="sec-hdr-title">할인율별 예측 시뮬레이션</div></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(lc(sim['할인율(%)'],sim['예상 리뷰 수'],'할인율별 예상 리뷰 수','할인율(%)','예상 리뷰 수',
                       bx=int(best['할인율(%)']),fmt='int'), use_container_width=True, config=MB)
    cls,msg = cc_review(sim,best)
    st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)

with col2:
    st.plotly_chart(lc(sim['할인율(%)'],sim['예상 평점'],'할인율별 예상 평점','할인율(%)','예상 평점',
                       bx=int(best['할인율(%)'])), use_container_width=True, config=MB)
    cls,msg = cc_rating(sim,best)
    st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(bc(sim['할인율(%)'],sim['추천점수'],int(best['할인율(%)'])),
                    use_container_width=True, config=MB)
    cls,msg = cc_score(best)
    st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)

with col4:
    st.plotly_chart(sc(df, price, np.log1p(best['예상 리뷰 수'])),
                    use_container_width=True, config=MB)
    cls,msg = cc_scatter(pp,rp)
    st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)

st.divider()


# =========================================================
# ④ 소비자 리뷰 반응 분석 (ABSA → 판매자 친화 명칭)
# =========================================================
st.markdown('<div class="sec-hdr"><div class="sec-hdr-icon">💬</div><div class="sec-hdr-title">소비자 리뷰 반응 분석</div></div>', unsafe_allow_html=True)

ca, cb = st.columns([3, 2])
with ca:
    fig_a = ac(absa_sum)
    if fig_a:
        st.plotly_chart(fig_a, use_container_width=True, config=MB)
        st.markdown(absa_table_html(absa_sum), unsafe_allow_html=True)
    else:
        st.info("분석에 충분한 리뷰 텍스트가 없습니다.")

with cb:
    st.plotly_chart(kc(kwdf), use_container_width=True, config=MB)
    st.markdown(kw_table_html(kwdf), unsafe_allow_html=True)

# 리뷰 반응 한 줄 요약
st.markdown(f'<div class="review-summary">{make_review_summary(top_asp, kwdf)}</div>', unsafe_allow_html=True)

st.divider()


# =========================================================
# ⑤ 최종 판매 전략 리포트
# =========================================================
st.markdown('<div class="sec-hdr"><div class="sec-hdr-icon">💡</div><div class="sec-hdr-title">최종 판매 전략 리포트</div></div>', unsafe_allow_html=True)

s1, s2 = st.columns(2)
with s1:
    st.markdown(f"""
    <div class="strat-card-blue">
      <div class="strat-label">할인 전략</div>
      <div class="strat-title">📌 {int(best['할인율(%)'])}% 할인 · {dsn}</div>
      <div class="strat-body">{dsb}</div>
    </div>
    """, unsafe_allow_html=True)
with s2:
    st.markdown(f"""
    <div class="strat-card-green">
      <div class="strat-label">상세페이지 전략</div>
      <div class="strat-title">🖊 {atn}</div>
      <div class="strat-body">{atb}</div>
    </div>
    """, unsafe_allow_html=True)

# 시장 포지션 수치
st.markdown(f"""
<div style="background:#fff;border:1px solid #E5E7EB;border-radius:12px;
            padding:20px 24px;margin-top:12px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
  <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
              letter-spacing:1.2px;color:#9CA3AF;margin-bottom:14px;">시장 내 위치 분석</div>
  <div style="display:flex;gap:40px;flex-wrap:wrap;margin-bottom:14px;">
    <div><div class="pos-lbl">가격 상위</div>
         <div class="pos-num">{pp:.1f}<span style="font-size:1.1rem;font-weight:600;color:#9CA3AF;">%</span></div></div>
    <div><div class="pos-lbl">예상 리뷰 상위</div>
         <div class="pos-num">{rp:.1f}<span style="font-size:1.1rem;font-weight:600;color:#9CA3AF;">%</span></div></div>
    <div><div class="pos-lbl">예상 평점 상위</div>
         <div class="pos-num">{rtp:.1f}<span style="font-size:1.1rem;font-weight:600;color:#9CA3AF;">%</span></div></div>
  </div>
  <div style="font-size:0.84rem;color:#6B7280;line-height:1.7;
              padding-top:12px;border-top:1px solid #F3F4F6;">
    추천 할인율 <strong style="color:#4F46E5">{int(best['할인율(%)'])}%</strong> 적용 시
    예상 평점 <strong style="color:#111827">{best['예상 평점']:.1f}점</strong>,
    예상 리뷰 <strong style="color:#111827">{int(best['예상 리뷰 수']):,}개</strong>로 예측됩니다.
    <span style="color:#9CA3AF;">(추천점수 산정: 평점 30% + 리뷰 수 30% + 매출 지표 40%)</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# =========================================================
# ⑥ 유사 상품 상세표 (접어두기)
# =========================================================
with st.expander("🔍 유사 상품 TOP 15 보기"):
    show = sim15[['product_name','display_category','actual_price',
                  'discount_percentage','rating','rating_count','similarity_score']].copy()
    show.columns = ['상품명','카테고리','가격','할인율(%)','평점','리뷰 수','유사도']
    st.dataframe(show.reset_index(drop=True), use_container_width=True)

with st.expander("📋 할인율별 전체 시뮬레이션 테이블"):
    disp = sim[['할인율(%)','할인 후 가격','예상 평점','예상 리뷰 수','추천점수']].copy()
    disp['예상 리뷰 수'] = disp['예상 리뷰 수'].astype(int)
    best_d = int(best['할인율(%)'])

    td = "padding:10px 14px;text-align:left;color:#111827;border-bottom:1px solid #F3F4F6;"
    th = ("padding:10px 14px;text-align:left;font-size:0.7rem;font-weight:700;"
          "text-transform:uppercase;letter-spacing:0.8px;color:#9CA3AF;border-bottom:2px solid #E5E7EB;")

    rows = ""
    for _, r in disp.iterrows():
        is_best = int(r['할인율(%)']) == best_d
        bg = "background:#F5F3FF;" if is_best else ""
        badge = (' <span style="background:#4F46E5;color:white;border-radius:4px;'
                 'padding:1px 7px;font-size:0.7rem;font-weight:700;margin-left:6px;">추천</span>'
                 if is_best else "")
        rows += (f'<tr style="{bg}">'
                 f'<td style="{td}">{int(r["할인율(%)"])}%{badge}</td>'
                 f'<td style="{td}">₹{int(r["할인 후 가격"]):,}</td>'
                 f'<td style="{td}">{r["예상 평점"]:.1f}</td>'
                 f'<td style="{td}">{int(r["예상 리뷰 수"]):,}개</td>'
                 f'<td style="{td}">{r["추천점수"]:.1f}점</td>'
                 f'</tr>')

    st.markdown(f"""
    <div style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;font-size:0.87rem;font-family:'Inter',sans-serif;">
      <thead><tr>
        <th style="{th}">할인율</th>
        <th style="{th}">할인 후 가격 (₹)</th>
        <th style="{th}">예상 평점</th>
        <th style="{th}">예상 리뷰 수</th>
        <th style="{th}">추천점수</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)
