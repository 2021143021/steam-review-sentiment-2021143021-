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
```

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

## 4. 탐색적 데이터 분석(EDA)

EDA는 `src/eda/plot_figures.py`에서 수행하며, 그 결과를 PNG 이미지로 저장해 본 보고서에서 활용한다.

### 4.1 라벨(긍/부정) 분포

`images/label_distribution.png` 그래프를 통해 라벨의 분포를 확인한다.

![라벨 분포](images/label_distribution.png)

- 대부분의 리뷰가 **추천(긍정, 1)**에 해당한다.
- 대략 **3:1** 정도의 비율로 긍정 리뷰가 더 많다.
- 이는 실제 게임 리뷰 환경에서 불만이 있어도 여전히 게임을 즐기는 유저가 많다는 점과, 일부는 단순 정보 공유용 리뷰를 남기기도 한다는 특성을 반영한다.
- 학습 시에는 이러한 **라벨 불균형(Imbalance)**을 인지하고 결과를 해석해야 한다.

### 4.2 리뷰 길이 분포

`images/review_length_hist.png` 그래프를 통해 리뷰 텍스트의 길이 분포를 확인한다.

![리뷰 길이 분포](images/review_length_hist.png)

- 리뷰 길이는 대체로 **100자 이내**의 짧은 리뷰가 많다.
- 0~400자 구간을 중심으로 히스토그램을 그리면, 짧은 리뷰가 급격히 많이 나타난다.

이 분포를 바탕으로 다음과 같은 전처리 기준을 수립한다.
- 의미 있는 텍스트를 확보하기 위해 너무 짧은 리뷰는 제거한다.
- 지나치게 긴 리뷰는 소수에 해당하므로, 모델 입력 길이를 **256 토큰** 정도로 제한해도 전체 데이터 대부분을 커버할 수 있다.

### 4.3 게임(app_id)별 리뷰 개수

`images/reviews_per_game.png` 그래프를 통해 게임별 데이터 분포를 확인한다.

![게임별 리뷰 개수](images/reviews_per_game.png)

- 일부 게임(`app_id`)에 리뷰가 집중되는 편이다.
- 예를 들어 매우 인기 있는 작품은 수천 개 이상의 리뷰를 가지고, 다른 게임들은 상대적으로 적은 리뷰를 가진다.

이 분포를 통해 다음을 확인한다.
- 데이터가 **특정 게임에 편향(Bias)**되어 있다.
- 이후 결과 해석 시, 특정 게임의 의견이 전체 감성 분포에 큰 영향을 줄 수 있음을 고려한다.

## 5. 모델 학습: KOELECTRA-small

### 5.1 모델 및 환경

- **사전학습 언어모델**: `monologg/koelectra-small-v3-discriminator`
- **작업**: 문장 단위 이진 분류 (긍정/부정)
- **구현**:
  - `transformers` 라이브러리의 `ElectraForSequenceClassification`을 사용한다.
  - **토크나이저**: `ElectraTokenizer`
- **입력 텍스트**:
  - Steam 리뷰 문자열
  - **최대 길이**: `max_length=256` 토큰
  - **패딩**: `padding="max_length"`
  - **트렁케이션**: `truncation=True`

### 5.2 데이터 분할 및 DataLoader

- **데이터 분할**: 전체 전처리 데이터(11,195건)를 기준으로 **Train:Validation = 8:2** 비율로 나눈다.
- **입력 텐서**:
  - `input_ids` (Int64)
  - `attention_mask`
  - `labels` (0/1, LongTensor)
- **배치 사이즈**: `batch_size = 32`
- **DataLoader**:
  - **Train**: `RandomSampler`
  - **Validation**: `SequentialSampler`

### 5.3 학습 하이퍼파라미터

- **Optimizer**: Adam
  - `lr = 1e-4`
  - `eps = 1e-6`
- **스케줄러**: `get_linear_schedule_with_warmup`
  - `num_warmup_steps = 0`
  - `num_training_steps = len(train_dataloader) * num_epochs`
- **Epoch 수**: 4
- **Gradient clipping**: `max_grad_norm = 1.0`
- **Seed**: `2025`로 고정하여 재현성을 확보한다.

### 5.4 Epoch별 학습 결과

학교 컴퓨터에서 학습한 최종 결과는 다음과 같다.

| Epoch | Train Loss | Train Acc | Val Acc |
|:---:|:---:|:---:|:---:|
| 1 | 0.4362 | 0.8630 | 0.8328 |
| 2 | 0.3323 | 0.9034 | 0.8306 |
| 3 | 0.2508 | 0.9337 | 0.8342 |
| **4** | **0.1896** | **0.9474** | **0.8335** |

- **Train Loss**는 Epoch가 진행될수록 지속적으로 감소한다.
- **Train Accuracy**는 **0.86 → 0.95** 수준까지 상승한다.
- **Validation Accuracy**는 **0.83** 근처에서 안정적으로 유지되며, Epoch 3~4에서 큰 변동 없이 수렴하는 양상을 보인다.

### 5.5 학습 곡선 시각화

#### 5.5.1 Train Loss 그래프

`images/train_loss.png` 그래프를 통해 학습 손실 변화를 확인한다.

![Train Loss](images/train_loss.png)

- Epoch가 증가할수록 Train Loss가 꾸준히 감소한다.
- 이는 모델이 학습 데이터를 점점 더 잘 설명하고 있음을 의미한다.

#### 5.5.2 Train vs Validation Accuracy 그래프

`images/train_val_accuracy.png` 그래프를 통해 정확도 추이를 비교한다.

![Accuracy](images/train_val_accuracy.png)

- **Train Accuracy**는 Epoch마다 상승하여 **0.94** 이상까지 올라간다.
- **Validation Accuracy**는 약 **0.83** 내외에서 수렴한다.
- Train과 Val 사이의 간격이 너무 크지 않아서 극단적인 과적합 상태는 아니지만, Train이 조금 더 높은 쪽으로 치우친다.

## 6. 모델 평가: 2,500개 검증셋 성능

검증용 데이터셋 정보는 다음과 같다.
- **파일:** `data/labeled/steam_reviews_for_labeling_labeled_auto.csv`
- **샘플 수:** 2,500개
- **컬럼:** `app_id`, `review`, `label`, `text_len`, `manual_label`, `manual_label_backup`, `labeling_source`

`src/eval/eval_on_manual_labels.py`에서 다음과 같은 평가를 수행한다.

### 6.1 전체 정확도

```text
Accuracy: 0.9256

2,500개 리뷰 중 약 **92.56%**를 올바르게 분류한다.
실전 감성 분석 모델로 사용하기에 충분히 의미 있는 수준의 정확도이다.

### 6.2 Classification Report

| Class | Precision | Recall | F1-Score | Support |
|:---:|:---:|:---:|:---:|:---:|
| **0 (부정)** | 0.9010 | 0.8753 | 0.8880 | 842 |
| **1 (긍정)** | 0.9376 | 0.9511 | 0.9443 | 1,658 |
| **Accuracy** | | | **0.9256** | 2,500 |
| **Macro Avg** | 0.9193 | 0.9132 | 0.9161 | 2,500 |
| **Weighted Avg** | 0.9252 | 0.9256 | 0.9253 | 2,500 |

**해석:**
- **긍정(1)** 클래스에서 Precision과 Recall이 모두 높고, F1-Score가 **0.94** 정도이다.
- **부정(0)** 클래스에서도 F1-Score가 **0.8880**으로, 특정 클래스만 잘 맞추는 모델이 아니다.
- Macro/Weighted 평균이 모두 **0.92** 근처로 고르게 나온다.

### 6.3 Confusion Matrix

Confusion Matrix는 다음과 같다.

| | 예측 0 (부정) | 예측 1 (긍정) |
|:---:|:---:|:---:|
| **실제 0 (부정)** | **737** | 105 |
| **실제 1 (긍정)** | 81 | **1,577** |

- 실제 부정(0)인 리뷰 842개 중 **737개**를 맞추고, 105개를 긍정으로 잘못 분류한다.
- 실제 긍정(1)인 리뷰 1,658개 중 **1,577개**를 맞추고, 81개를 부정으로 잘못 분류한다.

**이 결과는 다음을 시사한다.**
- 모델이 전반적으로 긍정/부정을 모두 잘 구분한다.
- 다만 부정 리뷰 일부(105건)를 긍정으로 판단하는 경향이 존재한다. 이 부분은 뒤의 한계점에서 다시 논의한다.
