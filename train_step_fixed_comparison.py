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
)

# ==========================================
# 1. [설정] 환경 및 경로 설정
# ==========================================
DATA_DIR = r"C:\Users\uze\PycharmProjects\sprint12"
MODEL_NAME = "digit82/kobart-summarization"

DOMAINS = ["news", "editorial", "law"]

# [핵심 변경] 도메인별 tokenized_dataset 경로. law만 target_length=190 버전 사용
# (editorial/news는 target 128로 기존과 동일 -> 재토큰화 불필요)
DOMAIN_DATASET_DIR = {
    "news": os.path.join(DATA_DIR, "tokenized_dataset_news"),
    "editorial": os.path.join(DATA_DIR, "tokenized_dataset_editorial"),
    "law": os.path.join(DATA_DIR, "tokenized_dataset_law_target190"),
}

# [핵심 변경] 가설 B 검증: 도메인당 샘플 수를 이 리스트 순서대로 전부 실험.
# law 실제 train 보유량이 24,099건이라, 그 이내로만 설정 (중복 없이 1:1:1 비율 유지)
SAMPLE_SIZES = [5000, 10000, 20000, 24000]

# [핵심 변경] 데이터량과 무관하게 모든 실험에서 동일하게 고정할 총 학습 스텝 수.
# 기존 5000개×3도메인 3epoch 실험의 총 step 수를 확인했다면 그 값으로 교체 권장.
FIXED_MAX_STEPS = 2800

VALID_SAMPLES_PER_DOMAIN = 300

EVAL_SAVE_STEPS = FIXED_MAX_STEPS // 5  # 총 5회 평가/저장

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 Device: {device}")
if device == "cuda":
    print(f"GPU 모델명: {torch.cuda.get_device_name(0)}")


# ==========================================
# 2. ROUGE 평가 함수 설정 (Mecab 형태소 지원)
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

    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # [핵심 추가] 생성 길이 자체를 추적 -> 가설 B의 핵심 관찰 지표
    gen_lens = [len(tokenizer(p)["input_ids"]) for p in decoded_preds]

    if use_mecab:
        decoded_preds = [" ".join(mecab.morphs(pred.strip())) for pred in decoded_preds]
        decoded_labels = [" ".join(mecab.morphs(label.strip())) for label in decoded_labels]

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        rouge_types=["rouge1", "rouge2", "rougeL"],
    )
    result = {k: round(v * 100, 4) for k, v in result.items()}
    result["gen_len_mean"] = round(float(np.mean(gen_lens)), 2)
    return result


# ==========================================
# 3. 도메인별 데이터셋 선택 추출 및 병합 함수
# ==========================================
def load_and_merge_domain_datasets(domains, train_per_domain, valid_per_domain):
    train_subsets = []
    valid_subsets = []

    print("\n도메인별 토크나이즈 데이터셋 추출 및 병합 중...")
    for dom in domains:
        dom_dir = DOMAIN_DATASET_DIR[dom]
        if not os.path.exists(dom_dir):
            raise FileNotFoundError(
                f"경로를 찾을 수 없습니다: {dom_dir}. law는 tokenize_law_target190.py를 먼저 실행하세요!"
            )

        ds_dict = load_from_disk(dom_dir)
        ds_train = ds_dict["train"]
        ds_valid = ds_dict["validation"]

        # [핵심] 요청량이 보유량을 넘지 않도록 캡 (law 24,099건 한계 -> 중복 없이 비율 유지)
        t_size = min(train_per_domain, len(ds_train))
        if t_size < train_per_domain:
            print(f" [{dom.upper()}] 요청량({train_per_domain:,}) > 보유량({len(ds_train):,}) -> {t_size:,}건으로 축소")
        sampled_train = ds_train.shuffle(seed=42).select(range(t_size))
        train_subsets.append(sampled_train)

        v_size = min(valid_per_domain, len(ds_valid))
        sampled_valid = ds_valid.shuffle(seed=42).select(range(v_size))
        valid_subsets.append(sampled_valid)

        print(f" └─> [{dom.upper()}] Train: {t_size:,}건 / Valid: {v_size:,}건 샘플링 완료")

    merged_train = concatenate_datasets(train_subsets).shuffle(seed=42)
    merged_valid = concatenate_datasets(valid_subsets).shuffle(seed=42)

    return merged_train, merged_valid


# ==========================================
# 4. 데이터량별 반복 실행 (가설 B: 스텝 수 고정 비교)
# ==========================================
if __name__ == "__main__":
    summary_log_path = os.path.join(DATA_DIR, "results_step_fixed", "summary_log.csv")
    os.makedirs(os.path.dirname(summary_log_path), exist_ok=True)

    for TRAIN_SAMPLES_PER_DOMAIN in SAMPLE_SIZES:
        run_name = f"step_fixed_{TRAIN_SAMPLES_PER_DOMAIN}perDomain"
        OUTPUT_DIR = os.path.join(DATA_DIR, "results_step_fixed", run_name)
        FINAL_MODEL_DIR = os.path.join(DATA_DIR, "final_models_step_fixed", run_name)

        print("\n" + "=" * 80)
        print(f"[{run_name}] 도메인당 {TRAIN_SAMPLES_PER_DOMAIN:,}건 / max_steps={FIXED_MAX_STEPS} 고정 학습 시작")
        print("=" * 80)

        print(f"모델 로드 중: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        # [핵심 추가] 그리드서치로 검증된 generation 설정을 학습 중 eval에도 동일하게 적용
        # (기존 90k 실험은 generation_max_length=64로 eval해서 best epoch 선택 자체가 왜곡됐었음)
        model.generation_config.max_length = 100
        model.generation_config.num_beams = 4
        model.generation_config.length_penalty = 0.8
        model.generation_config.no_repeat_ngram_size = 3

        train_dataset, eval_dataset = load_and_merge_domain_datasets(
            DOMAINS,
            train_per_domain=TRAIN_SAMPLES_PER_DOMAIN,
            valid_per_domain=VALID_SAMPLES_PER_DOMAIN,
        )

        print(f"\n최종 병합된 학습 데이터 수: {len(train_dataset):,}건 | 검증 데이터 수: {len(eval_dataset):,}건")

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        training_args = Seq2SeqTrainingArguments(
            output_dir=OUTPUT_DIR,
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=EVAL_SAVE_STEPS,
            save_steps=EVAL_SAVE_STEPS,

            learning_rate=2e-5,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            weight_decay=0.01,

            per_device_train_batch_size=4,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=4,  # 실질 배치 크기 = 16 (모든 실험 동일)
            dataloader_num_workers=0,

            predict_with_generate=True,
            generation_max_length=100,   # 64였던 기존 설정을 100으로 (조기 캡 문제 방지)
            generation_num_beams=4,

            # [핵심] epoch 대신 step 고정. 데이터가 적으면 여러 epoch 반복,
            # 많으면 1 epoch도 못 채우고 끝날 수 있음 -> 의도된 설계.
            max_steps=FIXED_MAX_STEPS,

            bf16=True if device == "cuda" else False,
            fp16=False,
            logging_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="rougeL",
            greater_is_better=True,
            report_to="tensorboard",
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        print("\n" + "=" * 80)
        print(f"{len(train_dataset):,}건 데이터로 도메인 균등 학습을 시작합니다! (max_steps={FIXED_MAX_STEPS})")
        print("=" * 80)
        train_result = trainer.train()

        trainer.log_metrics("train", train_result.metrics)

        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        print(f"\n[{run_name}] 최종 평가 지표: {eval_metrics}")

        trainer.save_model(FINAL_MODEL_DIR)
        tokenizer.save_pretrained(FINAL_MODEL_DIR)
        print(f"[{run_name}] 학습 완료! 최종 모델 경로: {FINAL_MODEL_DIR}")

        # 실험별 핵심 지표를 누적 기록 -> 나중에 데이터량별 추이를 한눈에 비교
        header_needed = not os.path.exists(summary_log_path)
        with open(summary_log_path, "a", encoding="utf-8-sig") as f:
            if header_needed:
                f.write("train_per_domain,total_docs,max_steps,rouge1,rouge2,rougeL,gen_len_mean\n")
            f.write(
                f"{TRAIN_SAMPLES_PER_DOMAIN},{len(train_dataset)},{FIXED_MAX_STEPS},"
                f"{eval_metrics.get('eval_rouge1', '')},"
                f"{eval_metrics.get('eval_rouge2', '')},"
                f"{eval_metrics.get('eval_rougeL', '')},"
                f"{eval_metrics.get('eval_gen_len_mean', '')}\n"
            )

    print("\n" + "=" * 80)
    print("모든 데이터량 실험 완료!")
    print(f"결과 비교: {summary_log_path}")
    print("=" * 80)
