![header](https://capsule-render.vercel.app/api?type=soft&color=auto&height=260&section=header&text=Steam%20Review%20Sentiment%20💬&fontSize=70)

# 🎮 Steam 한국어 게임 리뷰 감성 분석 프로젝트
**KOELECTRA-small 기반 긍·부정 분류 및 리뷰 패턴 분석**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-FF9A00?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![KOELECTRA](https://img.shields.io/badge/KOELECTRA-small-v3-00599C?style=flat-square&logo=googlecloud&logoColor=white)](https://huggingface.co/monologg/koelectra-small-v3-discriminator)

---

## 0. 목차

1. [프로젝트 개요](#1-프로젝트-개요)  
2. [데이터 수집](#2-데이터-수집)  
3. [데이터 전처리 및 라벨링](#3-데이터-전처리-및-라벨링)  
4. [탐색적 데이터 분석(EDA)](#4-탐색적-데이터-분석eda)  
5. [모델 학습: KOELECTRA-small](#5-모델-학습-koelectra-small)  
6. [모델 평가: 2,500개 검증셋 성능](#6-모델-평가-2500개-검증셋-성능)  
7. [리뷰 패턴 및 시사점](#7-리뷰-패턴-및-시사점)  
8. [한계점 및 향후 보완 방향](#8-한계점-및-향후-보완-방향)  
9. [코드 구조 및 재현 방법](#9-코드-구조-및-재현-방법)  
10. [그래프/이미지 파일 요약](#10-그래프이미지-파일-요약)  

---

## 1. 프로젝트 개요

### 1.1 문제 정의 및 동기

Steam 플랫폼에는 다양한 한국어 게임 리뷰가 쌓여 있고, 사용자들은 추천 여부(Recommend) 버튼과 함께 자유롭게 의견을 작성한다.  
그러나 리뷰 수가 매우 많기 때문에, 사람이 하나하나 읽으면서 **전체 분위기(긍정/부정)** 를 파악하기는 어렵다.  

본 프로젝트는 다음과 같은 문제의식에서 출발한다.

- 특정 게임에 대해 한국 유저들은 전반적으로 만족하는가, 불만족하는가  
- 부정 리뷰에서는 주로 어떤 이슈(버그, 최적화, 과금, 밸런스 등)가 반복되는가  
- **추천 여부 버튼만으로는 포착되지 않는 감정 구조**를 텍스트 기반 감성 분석으로 보완할 수 있는가  

이를 위해 한국어 사전학습 언어모델인 **KOELECTRA-small**을 활용하여  
Steam 한국어 리뷰를 **이진 감성 분류(긍정/부정)** 문제로 학습하고,  
EDA와 후속 분석을 통해 리뷰 패턴을 정량적으로 탐색한다.

### 1.2 프로젝트 목표

- Steam 한국어 리뷰 데이터를 직접 크롤링하여 **2만 건 이상** 수집한다.  
- KOELECTRA-small 기반 **문장 분류 모델**을 학습하여 추천 여부(label)를 예측한다.  
- 2,500개 샘플에 대해 별도의 라벨 기준을 만들고, 모델 성능을 정량적으로 평가한다.  
- 그래프 및 표를 활용하여:
  - 데이터 분포와 전처리 과정을 시각적으로 설명한다.  
  - Epoch별 학습/검증 성능을 그래프로 제시한다.  
- 실험 결과를 바탕으로 **게임 리뷰 감성 분석의 한계와 보완 방향**을 논의한다.

### 1.3 사용 데이터 개요

- 데이터 출처: **Steam 스토어 리뷰 페이지**  
- 언어: **한국어 리뷰** 중심  
- 총 수집 리뷰 수: **20,596건**  
- 라벨:
  - `recommend` (True/False) → `label` (1/0)로 변환  
  - 1: 추천(긍정), 0: 비추천(부정)  

자세한 스키마는 아래 전처리 섹션에서 정리한다.

---

## 2. 데이터 수집

### 2.1 수집 대상 게임 (app_id 및 커버 이미지)

본 프로젝트에서는 다음 **5개 Steam 게임**의 한국어 리뷰를 수집한다.

> 실제로 사용한 게임을 기준으로 app_id와 파일명을 맞추어 수정한다.

| app_id  | 게임 이름        | 커버 이미지 예시 |
|--------:|------------------|------------------|
| 1091500 | Cyberpunk 2077   | <img src="images/cyberpunk_2077.jpg" width="140"> |
| 1145360 | Hades            | <img src="images/hades.jpg" width="140">         |
| 1245620 | ELDEN RING       | <img src="images/elden_ring.jpg" width="140">   |
| 1623730 | Palworld         | <img src="images/palworld.jpg" width="140">     |
| 1627720 | Lies of P        | <img src="images/lies_of_p.jpg" width="140">    |

> 게임 커버 이미지는 Steam 스토어 페이지에서 가져온 것으로,  
> 교육용 학습 프로젝트 README에서만 사용한다.

각 게임은 장르와 플레이 스타일이 다르기 때문에,  
리뷰 양상도 서로 다르게 나타난다.  
이를 하나의 통합 코퍼스로 모아 **게임 전반의 감성 패턴**을 보고자 한다.

### 2.2 크롤링 파이프라인 구조

크롤링 코드는 `src/crawler/steam_review_crawler.py`에 위치한다.  

주요 흐름은 다음과 같다.

1. **입력**: app_id 목록, 페이지 수, 언어 필터(ko)  
2. Steam 리뷰 API 혹은 HTML 페이지 요청 (requests + pagination)  
3. 응답 JSON/HTML에서 다음 정보를 추출:
   - `review` (리뷰 본문)  
   - `recommend` (추천 여부, True/False)  
   - `timestamp_created` (UNIX time)  
   - `app_id` (게임 식별자)  
4. 리뷰를 누적하여 **게임별 CSV** 파일로 저장  
5. 이후 `data/raw/steam_reviews_raw.csv`로 병합하여 통합 데이터셋을 구성

실제 수집 결과는 다음과 같은 형태로 저장한다.

```text
data/raw/steam_reviews_raw.csv

id, app_id, recommend, review, timestamp_created, label
...

## 3. 데이터 전처리 및 라벨링

### 3.1 원본 스키마 및 기본 전처리

원본 CSV의 주요 컬럼은 다음과 같다.

- `id`: 리뷰 고유 ID
- `app_id`: 게임 식별자
- `recommend`: Steam 추천 여부(True/False)
- `review`: 리뷰 본문(자유 텍스트)
- `timestamp_created`: 작성 시각(UNIX time)
- `label`: `recommend`를 1/0으로 변환한 정수 라벨

기본 전처리 단계는 다음과 같다 (`src/preprocess/prepare_labeling_dataset.py`):

1. **결측치 제거**
   - `review` 또는 `label`이 비어 있는 행 제거
2. **텍스트 길이 계산**
   - `text_len` = len(review) 컬럼 추가
3. **너무 짧은 리뷰 제거**
   - 의미 있는 의견을 포함하지 않을 가능성이 높은 리뷰를 제거한다.
   - 예: 단순 이모티콘, 한 글자, "굿" 등 극단적으로 짧은 텍스트 일부 필터링

실제 로그 기준으로:
- 원본 리뷰 수: **20,596개**
- 길이 필터 적용 후: **11,195개**

이 전처리는 학습 데이터 품질을 조금이라도 높이기 위한 최소한의 규칙이다.

### 3.2 학습 데이터 추출 기준 (2,500개 샘플링)

과제 조건상 직접 라벨링 또는 검증에 사용할 데이터 2,000건 이상을 요구한다.
본 프로젝트에서는 다음과 같이 **2,500개 샘플**을 추출한다.

- 전체 전처리 후 리뷰: 11,195개
- 이 중에서 랜덤 샘플링으로 **2,500개** 선택
- 결과를 `data/labeled/steam_reviews_for_labeling.csv` 로 저장

해당 CSV의 주요 컬럼은 다음과 같다.
- `app_id`
- `review`
- `label` (추천 여부 기반 0/1)
- `text_len`
- `manual_label` (검증용 라벨 필드)

이 파일은 이후 자동/수동 라벨 확인과 모델 평가에 사용한다.

### 3.3 자동 라벨링 파이프라인 (KOELECTRA 기반)

추가로, 학습된 모델을 활용해 2,500개 샘플에 대한 자동 라벨링 결과를 생성한다.
관련 코드는 `src/labeling/auto_label_from_steam.py` 이다.

흐름은 다음과 같다.

1. `steam_reviews_for_labeling.csv` 로드
2. 이미 존재하는 `manual_label` 컬럼이 있다면 백업:
   - `manual_label_backup` 컬럼에 복사
3. 학습된 모델(`models/koelectra_steam`)을 로드하여 각 리뷰에 대해 0/1 예측을 수행
4. 예측 라벨을 `manual_label` 컬럼에 덮어쓴다.
5. 결과를 `steam_reviews_for_labeling_labeled_auto.csv` 로 저장한다.

로그 상 요약:
- **입력**: 2,500개
- **출력**: 2,500개, 컬럼 수 7개
  - `manual_label_backup`
  - `labeling_source` (라벨 출처 정보) 등 포함

이 데이터는 모델 성능 평가 및 라벨 품질 확인에 사용한다.
