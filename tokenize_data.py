import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# ==========================================
# 1. [설정] 본인의 로컬 환경 및 모델 설정
# ==========================================
DATA_DIR = r"C:\Users\uze\PycharmProjects\sprint12"
MODEL_NAME = "gogamza/kobart-base-v2"

TRAIN_CSV = os.path.join(DATA_DIR, "cleaned_dataset_train.csv")
VALID_CSV = os.path.join(DATA_DIR, "cleaned_dataset_valid.csv")

MAX_INPUT_LENGTH = 768
MAX_TARGET_LENGTH = 128

# 전역에서 tokenizer 선언 (Windows 멀티프로세싱 직렬화 에러 방지)
print(f"토크나이저 로드 중: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ==========================================
# 2. 토크나이징 함수 정의
# ==========================================
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["article"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        text_target=examples["summary"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length"
    )

    # Label 패딩 토큰을 -100으로 치환하여 Loss 연산 제외
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ==========================================
# 3. 메인 실행부
# ==========================================
if __name__ == "__main__":
    # 1) 전체 CSV 파일 읽기
    print("전체 CSV 파일 읽는 중...")
    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig", low_memory=False)
    df_valid = pd.read_csv(VALID_CSV, encoding="utf-8-sig", low_memory=False)

    # 결측치 방지를 위해 string(문자열) 타입으로 확실히 변환
    for df in [df_train, df_valid]:
        df["article"] = df["article"].fillna("").astype(str)
        df["summary"] = df["summary"].fillna("").astype(str)

    # 도메인 컬럼명 자동 감지 ('domain' 또는 'category')
    domain_col = "domain" if "domain" in df_train.columns else "category"
    if domain_col not in df_train.columns:
        raise KeyError("CSV 파일 내에 카테고리를 구분할 'domain' 또는 'category' 컬럼이 없습니다.")

    categories = df_train[domain_col].unique()
    print(f"감지된 카테고리 목록: {list(categories)}\n")

    # 2) 카테고리별로 CSV 분리 저장 및 토크나이징 루프
    for cat in categories:
        print("=" * 80)
        print(f"[{cat.upper()} 도메인] 분리 저장 및 토크나이징 시작")
        print("=" * 80)

        # 도메인별 필터링
        df_train_cat = df_train[df_train[domain_col] == cat].reset_index(drop=True)
        df_valid_cat = df_valid[df_valid[domain_col] == cat].reset_index(drop=True)

        print(f" Train 개수: {len(df_train_cat):,}건 | Valid 개수: {len(df_valid_cat):,}건")

        # [요청사항] 카테고리별 개별 CSV로 따로 저장
        cat_train_csv = os.path.join(DATA_DIR, f"cleaned_dataset_train_{cat}.csv")
        cat_valid_csv = os.path.join(DATA_DIR, f"cleaned_dataset_valid_{cat}.csv")

        df_train_cat.to_csv(cat_train_csv, index=False, encoding="utf-8-sig")
        df_valid_cat.to_csv(cat_valid_csv, index=False, encoding="utf-8-sig")
        print(f" [{cat}] CSV 파일 저장 완료:\n    - {cat_train_csv}\n    - {cat_valid_csv}")

        # Hugging Face DatasetDict 변환
        raw_datasets_cat = DatasetDict({
            "train": Dataset.from_pandas(df_train_cat[["article", "summary"]]),
            "validation": Dataset.from_pandas(df_valid_cat[["article", "summary"]])
        })

        # 토크나이징 진행
        print(f" [{cat}] 토크나이징 진행 중...")
        tokenized_datasets_cat = raw_datasets_cat.map(
            preprocess_function,
            batched=True,
            remove_columns=["article", "summary"],
            num_proc=1
        )

        # 도메인별 독립 폴더에 디스크 저장
        cat_output_dir = os.path.join(DATA_DIR, f"tokenized_dataset_{cat}")
        tokenized_datasets_cat.save_to_disk(cat_output_dir)
        print(f" [{cat}] 토크나이즈된 데이터셋 저장 완료 -> {cat_output_dir}\n")

    print("=" * 80)
    print("모든 카테고리별 CSV 분리 저장 및 토크나이징이 완벽하게 끝났습니다!")
    print("=" * 80)
