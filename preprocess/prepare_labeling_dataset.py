import os
import pandas as pd
import numpy as np

# .../steam-review-sentiment/src/preprocess/prepare_labeling_dataset.py
# -> BASE_DIR = .../steam-review-sentiment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "steam_reviews_raw.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "labeled")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "steam_reviews_for_labeling.csv")

TEXT_COL = "review"
LABEL_COL = "label"
LANG_COL = "language"   # 있으면 쓰고, 없으면 전체 사용

N_SAMPLES = 2500
MIN_LEN = 10  # 너무 짧은 리뷰는 제거


def main():
    print(f"원본 데이터 로드: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    print("원본 데이터 행 수:", len(df))
    print("컬럼 목록:", list(df.columns))

    # 필수 컬럼 체크
    for col in [TEXT_COL, LABEL_COL]:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼이 없습니다: {col}")

    # language 컬럼이 있으면 한국어만, 없으면 전체 사용
    if LANG_COL in df.columns:
        before = len(df)
        df = df[df[LANG_COL] == "ko"]
        after = len(df)
        print(f"language == 'ko' 필터 적용: {before} -> {after}")
    else:
        print("language 컬럼이 없어 전체 데이터를 사용합니다.")

    # 텍스트 길이 기준 필터
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df["text_len"] = df[TEXT_COL].str.len()

    before = len(df)
    df = df[df["text_len"] >= MIN_LEN]
    after = len(df)
    print(f"길이 필터 적용: {before} -> {after}")

    # 샘플링
    if len(df) < N_SAMPLES:
        print(f"경고: 데이터가 {N_SAMPLES}개 미만이라 전체 {len(df)}개를 사용합니다.")
        sampled = df
    else:
        sampled = df.sample(n=N_SAMPLES, random_state=2025)
        print(f"총 {len(df)}개 중에서 {N_SAMPLES}개 샘플링")

    # 라벨링용 CSV 저장
    sampled = sampled[["app_id", TEXT_COL, LABEL_COL, "text_len"]].copy()
    sampled["manual_label"] = np.nan  # 나중에 사람이/모델이 채울 컬럼

    sampled.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"라벨링용 CSV 저장 완료: {OUTPUT_PATH}")
    print("shape:", sampled.shape)


if __name__ == "__main__":
    main()
