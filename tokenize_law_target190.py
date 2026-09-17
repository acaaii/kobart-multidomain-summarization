"""
law 도메인만 target_length=190으로 재토크나이징.

- editorial/news는 target_length가 기존(128)과 동일하므로 재작업 불필요
  -> 기존 tokenized_dataset_editorial, tokenized_dataset_news를 그대로 사용하면 됨.
- law는 이미 존재하는 도메인별 CSV(cleaned_dataset_{split}_law.csv)를 바로 읽어서 처리.
  (전체 합친 cleaned_dataset_{split}.csv를 다시 읽고 재분리하는 중복 작업 제거)
"""

import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

DATA_DIR = r"C:\Users\uze\PycharmProjects\sprint12"
MODEL_NAME = "gogamza/kobart-base-v2"

MAX_INPUT_LENGTH = 768
LAW_TARGET_LENGTH = 190

LAW_TRAIN_CSV = os.path.join(DATA_DIR, "cleaned_dataset_train_law.csv")
LAW_VALID_CSV = os.path.join(DATA_DIR, "cleaned_dataset_valid_law.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, f"tokenized_dataset_law_target{LAW_TARGET_LENGTH}")

print(f"토크나이저 로드 중: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["article"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        text_target=examples["summary"],
        max_length=LAW_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
    )
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__ == "__main__":
    for path in [LAW_TRAIN_CSV, LAW_VALID_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"파일 없음: {path}")

    print(f"law 도메인 CSV 읽는 중...\n - {LAW_TRAIN_CSV}\n - {LAW_VALID_CSV}")
    df_train = pd.read_csv(LAW_TRAIN_CSV, encoding="utf-8-sig", low_memory=False)
    df_valid = pd.read_csv(LAW_VALID_CSV, encoding="utf-8-sig", low_memory=False)

    for df in [df_train, df_valid]:
        df["article"] = df["article"].fillna("").astype(str)
        df["summary"] = df["summary"].fillna("").astype(str)

    print(f"Train: {len(df_train):,}건 | Valid: {len(df_valid):,}건")

    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(df_train[["article", "summary"]]),
        "validation": Dataset.from_pandas(df_valid[["article", "summary"]]),
    })

    print(f"토크나이징 중 (input={MAX_INPUT_LENGTH}, target={LAW_TARGET_LENGTH})...")
    tokenized = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=["article", "summary"],
        num_proc=1,
    )

    tokenized.save_to_disk(OUTPUT_DIR)
    print(f"저장 완료 -> {OUTPUT_DIR}")
    print("\neditorial/news는 기존 tokenized_dataset_editorial, tokenized_dataset_news를 그대로 쓰면 됩니다.")
