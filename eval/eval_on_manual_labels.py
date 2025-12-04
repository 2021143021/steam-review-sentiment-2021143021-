import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from transformers import ElectraTokenizer, ElectraForSequenceClassification, logging as hf_logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# .../steam-review-sentiment/src/eval/eval_on_manual_labels.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "labeled", "steam_reviews_for_labeling_labeled_auto.csv"
)
MODEL_DIR = os.path.join(BASE_DIR, "models", "koelectra_steam")  # ⚠️ 학교 모델 경로

MAX_LEN = 256
BATCH_SIZE = 32


def main():
    # 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("사용하는 장치: cuda")
        print(" -> GPU 이름:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("사용하는 장치: cpu")

    hf_logging.set_verbosity_error()

    # [1] 데이터 로드
    print(f"[1] 평가용 데이터 로드: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print("    컬럼 목록:", list(df.columns))
    print("    전체 행 수:", len(df))

    if "manual_label" not in df.columns:
        raise ValueError("manual_label 컬럼이 없습니다.")

    df = df.dropna(subset=["manual_label"])
    df["manual_label"] = df["manual_label"].astype(int)
    print("    사용 가능한 라벨링 샘플 수:", len(df))
    print("    manual_label 고유값:", sorted(df["manual_label"].unique()))

    texts = df["review"].astype(str).tolist()
    labels = df["manual_label"].values.astype(np.int64)

    # [2] 토크나이저 / 모델 로드
    print(f"[2] 토크나이저 / 모델 로드: {MODEL_DIR}")
    tokenizer = ElectraTokenizer.from_pretrained(MODEL_DIR)
    model = ElectraForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    # [3] 토큰화
    print("[3] 토큰화 중...")
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )
    input_ids = np.array(encoded["input_ids"])
    attention_masks = np.array(encoded["attention_mask"])

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(input_ids, attention_masks, labels_t)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE)

    # [4] 모델 예측
    print("[4] 모델 예측 및 평가 중...")

    all_preds = []
    all_true = []

    for batch in dataloader:
        batch_ids, batch_masks, batch_labels = (t.to(device) for t in batch)

        with torch.no_grad():
            outputs = model(batch_ids, attention_mask=batch_masks)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_true.extend(batch_labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    acc = accuracy_score(all_true, all_preds)
    print("\n=== 수동(자동) 라벨 기준 성능 평가 ===")
    print(f"Accuracy: {acc:.4f}\n")

    print("[Classification Report]")
    print(classification_report(all_true, all_preds, digits=4))

    print("\n[Confusion Matrix]")
    print(confusion_matrix(all_true, all_preds))


if __name__ == "__main__":
    main()
