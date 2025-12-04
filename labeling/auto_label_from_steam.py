import os
import pandas as pd
import numpy as np

# .../steam-review-sentiment/src/labeling/auto_label_from_steam.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_PATH = os.path.join(BASE_DIR, "data", "labeled", "steam_reviews_for_labeling.csv")
OUTPUT_PATH = os.path.join(
    BASE_DIR, "data", "labeled", "steam_reviews_for_labeling_labeled_auto.csv"
)

TEXT_COL = "review"
STEAM_LABEL_COL = "label"        # Steam 추천 여부
MANUAL_LABEL_COL = "manual_label"


def main():
    print(f"[1] 라벨링 대상 CSV 로드: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print("    컬럼 목록:", list(df.columns))
    print("    원본 행 수:", len(df))

    # manual_label 백업
    if MANUAL_LABEL_COL in df.columns:
        backup_col = MANUAL_LABEL_COL + "_backup"
        print(f"    경고: '{MANUAL_LABEL_COL}' 컬럼이 이미 존재합니다. '{backup_col}' 으로 백업합니다.")
        df[backup_col] = df[MANUAL_LABEL_COL]

    # Steam label -> manual_label 로 복사
    if STEAM_LABEL_COL not in df.columns:
        raise ValueError(f"'{STEAM_LABEL_COL}' 컬럼이 없습니다.")

    df[MANUAL_LABEL_COL] = df[STEAM_LABEL_COL].astype(int)
    print("    manual_label 고유값:", sorted(df[MANUAL_LABEL_COL].unique()))

    # 라벨 소스 메타 정보
    df["labeling_source"] = "from_steam_recommendation"

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print("[2] 자동 라벨링 완료")
    print("    저장 경로:", OUTPUT_PATH)
    print("    shape:", df.shape)


if __name__ == "__main__":
    main()
