# SellMate — AI 기반 판매자 고객 반응 예측 시스템

**판매자의 AI 파트너** · 할인율 최적화 · 평점 예측 · 소비자 반응 분석 · 판매 전략 자동 추천

**25-2-생성형AI 초보1팀**

---

## 프로젝트 개요

아마존 판매자가 상품 등록 전에 가장 고민하는 질문들이 있습니다.

> *"할인율은 얼마가 적당할까?"*  
> *"소비자들은 어떤 부분을 중요하게 생각할까?"*  
> *"비슷한 상품 대비 우리 상품의 위치는?"*

**SellMate**는 Amazon Sales Dataset을 기반으로 이 질문들에 데이터 기반으로 답하는 AI 판매 전략 대시보드입니다. GradientBoosting 예측 모델, TF-IDF 유사 상품 탐색, ABSA 감성 분석을 결합하여 수초 내에 최적 할인율과 상세페이지 전략을 자동으로 제안합니다.

---

## 핵심 기능

| 기능 | 설명 |
|------|------|
| **AI 할인율 추천** | 0~50% 구간 시뮬레이션으로 최적 할인율 자동 도출 |
| **평점 예측** | GradientBoosting 기반 예상 평점 제공 (R²=0.412) |
| **리뷰 수 예측** | ML 모델로 예상 리뷰 수 사전 예측 (R²=0.623) |
| **소비자 반응 분석** | 유사 상품 리뷰 ABSA 분석으로 속성별 감성 파악 |
| **유사 상품 탐색** | TF-IDF + Cosine Similarity 기반 TOP 15 유사 상품 제공 |
| **시장 포지션 시각화** | 전체 시장 대비 가격·리뷰 위치 산점도 |

---

## 전체 파이프라인

```
Amazon Dataset
    │
    ▼
전처리 (가격 기호 제거, 카테고리 파싱, 파생 변수 생성)
    │
    ▼
EDA (평점 분포, 할인율 분포, 카테고리 분석)
    │
    ├──────────────────────────┐
    ▼                          ▼
TF-IDF 유사상품 탐색         ML 모델 (GradientBoosting)
Cosine Similarity 계산       평점 예측 / 리뷰 수 예측
유사 상품 TOP 15 추출         할인율별 시뮬레이션
    │                          │
    └──────────┬───────────────┘
               ▼
    추천 할인율 + 예상 평점 + 예상 리뷰 수
               │
               ▼
    Streamlit 웹 서비스 (SellMate 대시보드)
```
<img width="1513" height="909" alt="image" src="https://github.com/user-attachments/assets/c8972a2e-ad5b-4350-b468-4d17c307a22b" />


---

## 모델 성능

| 모델 | 목표 | R² | RMSE | MAE |
|------|------|----|------|-----|
| Model A | 평점 예측 | 0.412 | 0.284 | 0.221 |
| Model B | 리뷰 수 예측 (log scale) | 0.623 | 0.891 | 0.672 |

리뷰 수는 가격·할인율과의 상관관계가 더 명확하여 Model B가 더 높은 성능을 기록했습니다.

**추천점수 산정 기준:** 예상 평점 30% + 예상 리뷰 수 30% + 예상 매출 지표 40%

**GradientBoosting 선택 이유:** 비선형 관계 포착에 우수하며, 범주형 변수 처리가 가능하고 과적합에 강인한 특성을 보여 실험 결과 최고 R² 성능을 기록했습니다.

---

## 폴더 구조

```
sellmate/
├── app.py                   # Streamlit 메인 앱 (데이터 로딩, 모델 학습, 시각화, UI 전체 포함)
├── amazon.csv               # Amazon Sales Dataset (별도 다운로드 필요)
├── requirements.txt
└── README.md
```

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| 웹 서비스 | Python, Streamlit |
| 머신러닝 | scikit-learn (GradientBoostingRegressor, Pipeline, ColumnTransformer) |
| 텍스트 분석 | TF-IDF Vectorizer, Cosine Similarity |
| 시각화 | Plotly (인터랙티브 차트) |
| 데이터 처리 | pandas, numpy |
| UI 스타일링 | Custom CSS |

---

## 실행 방법

**1. 의존성 설치**

```bash
pip install streamlit pandas numpy plotly scikit-learn
```

또는

```bash
pip install -r requirements.txt
```

**2. 데이터 파일 준비**

`amazon.csv` 파일을 프로젝트 루트 디렉토리에 위치시킵니다.

- 데이터 출처: [Kaggle — Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)
- 필수 컬럼: `product_name`, `actual_price`, `discount_percentage`, `rating`, `rating_count`, `category`, `about_product`, `review_title`, `review_content`

**3. 앱 실행**

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 로 접속합니다.

---

## 사용 시나리오

```
1. 상품 입력    →  판매 중 상품 검색(키워드) 또는 신상품 정보 직접 입력
2. AI 분석      →  분석 시작 버튼 클릭, 수초 내 결과 출력
3. 할인율 확인  →  최적 할인율 및 할인율별 시뮬레이션 차트 검토
4. 평점 확인    →  예상 평점·리뷰 수·시장 포지션 파악
5. 반응 분석    →  소비자가 중시하는 속성 파악 (ABSA 감성 분석)
6. 전략 확인    →  AI가 제안하는 상세페이지 전략 확인
7. 상품 등록    →  전략 적용 후 상품 최종 등록
```

---

## 데이터 소개

| 항목 | 내용 |
|------|------|
| 분석 상품 수 | 1,465+ |
| 주요 카테고리 | 8개 (Computer, Electronics, Home, Office 등) |
| 활용 컬럼 수 | 16개+ |

**전처리 내용**

원본 데이터의 가격(`₹`, `,` 기호), 할인율(`%` 기호), 리뷰 수(`,` 기호)를 제거하고 수치형으로 변환했습니다. 카테고리 컬럼은 `|` 구분자 기준 1단계 값만 추출했습니다. 분석에 활용하기 위해 `price_level`(카테고리 내 상대 가격대), `log_rating_count`, `price_discount_interaction`, `combined_text` 4개의 파생 변수를 추가로 생성했습니다.

**핵심 EDA 인사이트**
- 대부분 상품의 평점은 3.5~4.5점 구간에 집중
- 할인율 20~30% 구간에 상품이 가장 많음
- 전자기기 카테고리가 전체의 약 63% 차지

---

## 분석 방법론

### TF-IDF 기반 유사 상품 탐색

상품명, 설명, 카테고리를 결합한 `combined_text`를 TF-IDF(max_features=8,000) 벡터로 변환합니다. 분석 대상 상품의 벡터와 전체 상품 벡터 간 Cosine Similarity를 계산하여 유사도 상위 15개 상품을 추출하고, 해당 상품들의 리뷰 텍스트를 ABSA 분석에 활용합니다.

### ABSA (Aspect-Based Sentiment Analysis)

가격/가성비, 품질, 배송/포장, 사용성, 디자인, 성능 6개 속성에 대해 각각 도메인 키워드 사전을 정의하고, 키워드 주변 윈도우(±3 토큰) 내에서 긍/부정 단어 빈도를 집계해 속성별 감성을 분류합니다. 결과는 속성별 언급 빈도와 긍·중립·부정 비율로 시각화됩니다.

### 할인율 시뮬레이션

0~50% 할인율 구간을 5% 단위로 나누어, 각 구간에서 Model A(평점)와 Model B(리뷰 수)를 순차적으로 예측합니다. 예측값을 min-max 정규화한 뒤 예상 매출 지표(할인 후 가격 × 예상 리뷰 수)와 함께 가중 합산하여 종합 추천점수(100점 만점)를 산출합니다.

---

## 한계 및 향후 발전 방향

**현재 한계**
- Amazon 인도 데이터만 활용 — 국내 쇼핑몰 직접 적용 불가
- 리뷰 텍스트 영어 기반 — 한국어 미지원
- 실시간 가격·리뷰 데이터 미반영
- 분석 가능 카테고리 8개로 한정
- 리뷰 수 예측 R²=0.623으로 설명력에 한계 존재

**향후 발전 방향**
1. **LLM 연동** — GPT 기반 상품 전략 문구 자동 생성
2. **실시간 크롤링** — 최신 리뷰·가격 데이터 실시간 반영
3. **국내 쇼핑몰 지원** — 쿠팡·네이버·11번가 데이터 확장
4. **AI 마케팅 문구** — 속성 기반 광고 카피 자동 생성
5. **판매량 예측** — 리뷰 수를 넘어 실제 매출 예측

---

*DACOS 초보1팀 · 2026*
