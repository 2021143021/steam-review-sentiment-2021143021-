import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import (
    ElectraTokenizer,
    ElectraForSequenceClassification,
    get_linear_schedule_with_warmup,
    logging,
)
from tqdm import tqdm

# =========================
# 0. 환경 설정
# =========================

# 이 파일 기준으로 프로젝트 루트 경로 계산
# .../steam-review-sentiment/src/training/train_koelectra_steam.py
# -> BASE_DIR = .../steam-review-sentiment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("사용하는 장치: cuda")
    print(" -> GPU 이름:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("사용하는 장치: cpu")

# transformers 경고 줄이기
logging.set_verbosity_error()

# 재현성 설정
SEED = 2025
np.random.seed(SEED)
torch.manual_seed(SEED)
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# =========================
# 1. 데이터 로드
# =========================

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "steam_reviews_raw.csv")
print("데이터 경로:", DATA_PATH)

df = pd.read_csv(DATA_PATH)

# review / label 컬럼만 사용
df = df[["review", "label"]].dropna(axis=0)

texts = df["review"].astype(str).tolist()
labels = df["label"].astype(int).values
labels = labels.astype(np.int64)  # 🔥 라벨을 long 타입으로 고정

print("### 데이터 확인 ###")
num_to_print = 3
for j in range(num_to_print):
    print(f"리뷰: {texts[j][:60]}...")
    print(f"\t라벨(추천여부 label): {labels[j]}")
print(f"\n\t* 전체 데이터 수: {len(texts)}")
print(f"\t* 부정(0) 리뷰 수: {list(labels).count(0)}")
print(f"\t* 긍정(1) 리뷰 수: {list(labels).count(1)}")
print("라벨 고유값:", np.unique(labels))

# =========================
# 2. 토크나이저 & 토큰화
# =========================

MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
# 로컬에 'koelectra-base-v3-discriminator' 폴더가 있으면 그걸 써도 됨.

tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)

encoded = tokenizer(
    texts,
    truncation=True,
    max_length=256,
    padding="max_length",
)

input_ids = np.array(encoded["input_ids"])
attention_masks = np.array(encoded["attention_mask"])

print("\n### 토큰화 결과 샘플 ###")
for j in range(num_to_print):
    print(f"\n{j+1}번째 데이터")
    print(" ## 토큰 ID ##")
    print(input_ids[j][:20], "...")
    print(" ## 어텐션 마스크 ##")
    print(attention_masks[j][:20], "...")

# =========================
# 3. train / validation 분리
# =========================

train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    input_ids,
    labels,
    test_size=0.2,
    random_state=SEED,
)

train_masks, val_masks, _, _ = train_test_split(
    attention_masks,
    labels,
    test_size=0.2,
    random_state=SEED,
)

# =========================
# 4. DataLoader 구성
# =========================

batch_size = 32

train_inputs = torch.tensor(train_inputs, dtype=torch.long)
train_labels = torch.tensor(train_labels, dtype=torch.long)
train_masks = torch.tensor(train_masks, dtype=torch.long)

train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(
    train_dataset,
    sampler=train_sampler,
    batch_size=batch_size,
    pin_memory=(device.type == "cuda"),
)

val_inputs = torch.tensor(val_inputs, dtype=torch.long)
val_labels = torch.tensor(val_labels, dtype=torch.long)
val_masks = torch.tensor(val_masks, dtype=torch.long)

val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_dataset)
val_dataloader = DataLoader(
    val_dataset,
    sampler=val_sampler,
    batch_size=batch_size,
    pin_memory=(device.type == "cuda"),
)

# =========================
# 5. 모델 / 옵티마이저 / 스케줄러
# =========================

model = ElectraForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)
model.to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    eps=1e-6,
    betas=(0.9, 0.999),
)

num_epochs = 4
total_steps = len(train_dataloader) * num_epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps,
)

epoch_results = []

# =========================
# 6. 학습 루프
# =========================

for epoch in range(num_epochs):
    print(f"\n===== Epoch {epoch + 1} / {num_epochs} =====")

    # ----- Training -----
    model.train()
    total_train_loss = 0.0

    train_progress = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}", leave=False)

    for batch in train_progress:
        batch_ids, batch_masks, batch_labels = (
            t.to(device, non_blocking=True) for t in batch
        )

        model.zero_grad()

        outputs = model(
            batch_ids,
            attention_mask=batch_masks,
            labels=batch_labels,
        )

        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_progress.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_train_loss = total_train_loss / len(train_dataloader)

    # ----- Train Accuracy -----
    model.eval()
    train_preds = []
    train_true = []

    for batch in tqdm(train_dataloader, desc=f"Evaluating Train Epoch {epoch + 1} (Train)", leave=False):
        batch_ids, batch_masks, batch_labels = (
            t.to(device, non_blocking=True) for t in batch
        )

        with torch.no_grad():
            outputs = model(batch_ids, attention_mask=batch_masks)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        train_preds.extend(preds.cpu().numpy())
        train_true.extend(batch_labels.cpu().numpy())

    train_accuracy = np.mean(np.array(train_preds) == np.array(train_true))

    # ----- Validation Accuracy -----
    val_preds = []
    val_true = []

    for batch in tqdm(val_dataloader, desc=f"Evaluating Validation Epoch {epoch + 1}", leave=False):
        batch_ids, batch_masks, batch_labels = (
            t.to(device, non_blocking=True) for t in batch
        )

        with torch.no_grad():
            outputs = model(batch_ids, attention_mask=batch_masks)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        val_preds.extend(preds.cpu().numpy())
        val_true.extend(batch_labels.cpu().numpy())

    val_accuracy = np.mean(np.array(val_preds) == np.array(val_true))

    epoch_results.append((avg_train_loss, train_accuracy, val_accuracy))

    print(
        f"\n[Epoch {epoch + 1}] "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

# =========================
# 7. 결과 요약
# =========================

print("\n==== 전체 Epoch 학습 결과 요약 ====")
for idx, (loss, train_acc, val_acc) in enumerate(epoch_results, start=1):
    print(
        f"Epoch {idx}: Train Loss={loss:.4f}, "
        f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}"
    )

# =========================
# 8. 모델 저장
# =========================

SAVE_DIR = os.path.join(BASE_DIR, "models", "koelectra_steam")

print("\n=== 모델 저장 중 ===")
os.makedirs(SAVE_DIR, exist_ok=True)

model.cpu()
for param in model.parameters():
    if not param.is_contiguous():
        param.data = param.data.contiguous()

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"=== 저장 완료: {SAVE_DIR} ===")
