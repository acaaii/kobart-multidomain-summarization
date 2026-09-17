import os
import torch
import numpy as np
import evaluate
from datasets import concatenate_datasets, load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

# ==========================================
# 1. [설정] 환경 및 경로 설정
# ==========================================
DATA_DIR = r"C:\Users\uze\PycharmProjects\sprint12"
MODEL_NAME = "digit82/kobart-summarization"

# [핵심 설정] 사용할 도메인 목록
DOMAINS = ["news", "editorial", "law"]

# [핵심 설정] 도메인별 샘플링 건수 (원하는 데이터 수로 수정하세요)
# 예: 3만 건 실험 시 각 10,000건 / 7만 건 실험 시 각 23,000건
TRAIN_SAMPLES_PER_DOMAIN = 30000

# [요청 반영] Validation 데이터는 도메인별 5,000건 고정 (총 15,000건)
VALID_SAMPLES_PER_DOMAIN = 300

# 결과 및 가중치를 분리 저장할 경로 (예: _30k_balanced)
OUTPUT_DIR = os.path.join(DATA_DIR, "results_kobart_summary_90k")
EPOCH_WEIGHTS_DIR = os.path.join(DATA_DIR, "saved_epoch_weights_90k")
FINAL_MODEL_DIR = os.path.join(DATA_DIR, "final_kobart_summary_model_90k")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 Device: {device}")
if device == "cuda":
    print(f"GPU 모델명: {torch.cuda.get_device_name(0)}")


# ==========================================
# 2. 커스텀 에폭 가중치 저장 콜백
# ==========================================
class SaveEpochWeightsCallback(TrainerCallback):
    def __init__(self, save_dir, tokenizer):
        self.save_dir = save_dir
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(state.epoch)
        epoch_save_path = os.path.join(self.save_dir, f"weights_epoch_{epoch}")
        os.makedirs(epoch_save_path, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"[Epoch {epoch} 종료] 모델 가중치 저장 -> {epoch_save_path}")
        print("=" * 60)

        if model is not None:
            model.save_pretrained(epoch_save_path)
            self.tokenizer.save_pretrained(epoch_save_path)


# ==========================================
# 3. ROUGE 평가 함수 설정 (Mecab 형태소 지원)
# ==========================================
rouge = evaluate.load("rouge")

try:
    from mecab import MeCab

    mecab = MeCab()
    use_mecab = True
except ImportError:
    use_mecab = False


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.where(
        predictions != -100, predictions, tokenizer.pad_token_id
    )
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True
    )
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    if use_mecab:
        decoded_preds = [
            " ".join(mecab.morphs(pred.strip())) for pred in decoded_preds
        ]
        decoded_labels = [
            " ".join(mecab.morphs(label.strip())) for label in decoded_labels
        ]

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        rouge_types=["rouge1", "rouge2", "rougeL"],
    )
    return {k: round(v * 100, 4) for k, v in result.items()}


# ==========================================
# 4. 도메인별 데이터셋 선택 추출 및 병합 함수
# ==========================================
def load_and_merge_domain_datasets(
    domains, train_per_domain, valid_per_domain
):
    train_subsets = []
    valid_subsets = []

    print("\n도메인별 토크나이즈 데이터셋 추출 및 병합 중...")
    for dom in domains:
        dom_dir = os.path.join(DATA_DIR, f"tokenized_dataset_{dom}")
        if not os.path.exists(dom_dir):
            raise FileNotFoundError(
                f"경로를 찾을 수 없습니다: {dom_dir}. 먼저 도메인별 토크나이징을 진행해 주세요!"
            )

        ds_dict = load_from_disk(dom_dir)
        ds_train = ds_dict["train"]
        ds_valid = ds_dict["validation"]

        # Train: 도메인 내에서 최대 지정 건수만큼 샘플링
        t_size = min(train_per_domain, len(ds_train))
        sampled_train = ds_train.shuffle(seed=42).select(range(t_size))
        train_subsets.append(sampled_train)

        # Valid: 요청대로 도메인별 5,000건(또는 해당 도메인 최대치) 추출
        v_size = min(valid_per_domain, len(ds_valid))
        sampled_valid = ds_valid.shuffle(seed=42).select(range(v_size))
        valid_subsets.append(sampled_valid)

        print(
            f" └─> [{dom.upper()}] Train: {t_size:,}건 / Valid: {v_size:,}건 샘플링 완료"
        )

    # 1:1:1 비율로 병합 및 최종 셔플
    merged_train = concatenate_datasets(train_subsets).shuffle(seed=42)
    merged_valid = concatenate_datasets(valid_subsets).shuffle(seed=42)

    return merged_train, merged_valid


# ==========================================
# 5. 메인 파인튜닝 실행부
# ==========================================
if __name__ == "__main__":
    print(f"모델 로드 중: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # 도메인별 추출 및 병합 실행
    train_dataset, eval_dataset = load_and_merge_domain_datasets(
        DOMAINS,
        train_per_domain=TRAIN_SAMPLES_PER_DOMAIN,
        valid_per_domain=VALID_SAMPLES_PER_DOMAIN,
    )

    print(
        f"\n최종 병합된 학습 데이터 수: {len(train_dataset):,}건 | 검증 데이터 수: {len(eval_dataset):,}건"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",

        # [핵심 LR 및 스케줄러 설정]
        learning_rate=2e-5,  # 기본 학습률
        warmup_ratio=0.1,  # 10% 예열
        lr_scheduler_type="cosine",  # Cosine Decay
        weight_decay=0.01,  # 가중치 감쇄

        # [배치 및 메모리/속도 최적화]
        per_device_train_batch_size=4,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,  # 실질 배치 크기 = 16
        dataloader_num_workers=0,  # <-- 추가: 데이터 피딩 속도 향상 (윈도우 추천: 2)

        # [검증 요약문 생성 제어]
        predict_with_generate=True,
        generation_max_length=64,  # <-- 수정: 128 -> 64 (과도한 문장 늘어짐 방지)
        generation_num_beams=2,  # <-- 추가: 안정적인 Beam Search 검증 (속도/품질 타협점)

        num_train_epochs=3,
        bf16=True if device == "cuda" else False,
        fp16=False,
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        report_to="tensorboard"  # <-- 수정: none -> tensorboard (포트폴리오 그래프 기록)
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            SaveEpochWeightsCallback(
                save_dir=EPOCH_WEIGHTS_DIR, tokenizer=tokenizer
            )
        ],
    )

    print("\n" + "=" * 80)
    print(
        f"{len(train_dataset):,}건 데이터로 도메인 균등 학습을 시작합니다!"
    )
    print("=" * 80)
    train_result = trainer.train()

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    print("\n" + "=" * 80)
    print(f"모든 학습 완료! 에폭별 가중치 저장 폴더: {EPOCH_WEIGHTS_DIR}")
    print(f"최종 최고 성능 모델 경로: {FINAL_MODEL_DIR}")
    print("=" * 80)
