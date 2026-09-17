"""
가설 B 최종 검증: max_steps 고정 재학습으로 나온 4개 모델(5000/10000/20000/24000)을
동일한 검증셋 + 동일한 generation 설정으로 평가하고 나란히 비교.

기존 evaluate 스크립트(ROUGE + BERTScore + NLI)와 check_generation_cap_ratio.py의
로직을 합쳐서, 모델별로 한 번에 다 계산합니다.

핵심 관찰 지표:
- gen_len_mean, domain별 gen_len_mean: 데이터량 늘어도 길이가 그대로면 가설 B 입증
- capped_ratio(%): max_length(128)에서 강제로 잘리는 비율. 데이터량과 무관하게 낮게
  유지되면, 기존 90k 모델에서 봤던 문제가 "학습 스텝 수" 때문이었다는 게 확정됨
"""

import os

# [디버깅용] CUDA 에러는 비동기로 보고되어 트레이스백 위치가 실제 원인과 다를 수 있음.
# 동기 실행으로 바꿔서 진짜 문제가 발생하는 지점을 정확히 찾는다.
# 원인 파악 후에는 속도를 위해 다시 지워도 됨.
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import pandas as pd
import numpy as np
import evaluate
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)

# ==========================================
# 1. 설정
# ==========================================
DATA_DIR = r"C:\Users\uze\PycharmProjects\sprint12"
DOMAINS = ["news", "editorial", "law"]

# 가설 B 실험으로 나온 4개 모델
MODEL_DIRS = {
    5000: os.path.join(DATA_DIR, "final_models_step_fixed", "step_fixed_5000perDomain"),
    10000: os.path.join(DATA_DIR, "final_models_step_fixed", "step_fixed_10000perDomain"),
    20000: os.path.join(DATA_DIR, "final_models_step_fixed", "step_fixed_20000perDomain"),
    24000: os.path.join(DATA_DIR, "final_models_step_fixed", "step_fixed_24000perDomain"),
}

EVAL_SAMPLES_PER_DOMAIN = 300
QUALITATIVE_SAVE_PER_DOMAIN = 10  # 모델별로 눈검수용 소량 저장

# 그리드서치로 검증된 generation 설정. law는 target_length=190으로 학습했으므로
# eval에서도 128로 다시 묶으면 방금 없앤 truncation 문제가 재발함 -> law만 여유 있게 200.
GEN_KWARGS_BY_DOMAIN = {
    "news": dict(max_length=128, min_length=15, num_beams=4, length_penalty=0.8,
                 no_repeat_ngram_size=3, early_stopping=True),
    "editorial": dict(max_length=128, min_length=15, num_beams=4, length_penalty=0.8,
                       no_repeat_ngram_size=3, early_stopping=True),
    "law": dict(max_length=200, min_length=15, num_beams=4, length_penalty=0.8,
                no_repeat_ngram_size=3, early_stopping=True),
}

SUMMARY_OUTPUT_CSV = os.path.join(DATA_DIR, "step_fixed_evaluation_summary.csv")

device = 0 if torch.cuda.is_available() else -1
device_name = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 Device: {device_name}")


# ==========================================
# 2. 평가 지표 준비
# ==========================================
print("평가 지표 및 보조 검증 모델 로드 중...")

rouge_metric = evaluate.load("rouge")
tokenizer_mode = "whitespace"
try:
    from mecab import MeCab
    mecab = MeCab()
    tokenizer_mode = "mecab"
    print(" [ROUGE] Mecab 형태소 분석기 적용")
except ImportError:
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        tokenizer_mode = "kiwi"
        print(" [ROUGE] Kiwi 형태소 분석기 적용 (Mecab 대체)")
    except ImportError:
        print(" [ROUGE] 형태소 분석기 미설치, 기본 띄어쓰기 기준 채점")

bertscore_metric = evaluate.load("bertscore")

NLI_MODEL_NAME = "Huffon/klue-roberta-base-nli"
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
nli_model = AutoModelForSequenceClassification.from_pretrained(
    NLI_MODEL_NAME, attn_implementation="eager"
)
if torch.cuda.is_available():
    nli_model = nli_model.to("cuda")
nli_model.eval()
print(" [BERTScore & NLI] 준비 완료")


def tokenize_for_rouge(text):
    text_strip = text.strip()
    if tokenizer_mode == "mecab":
        return " ".join(mecab.morphs(text_strip))
    elif tokenizer_mode == "kiwi":
        return " ".join([t.form for t in kiwi.tokenize(text_strip)])
    return text_strip


def calculate_rouge(predictions, references):
    preds_m = [tokenize_for_rouge(p) for p in predictions]
    refs_m = [tokenize_for_rouge(r) for r in references]
    results = rouge_metric.compute(
        predictions=preds_m, references=refs_m,
        rouge_types=["rouge1", "rouge2", "rougeL"],
    )
    return {k: round(v * 100, 2) for k, v in results.items()}


def calculate_bertscore(predictions, references):
    results = bertscore_metric.compute(
        predictions=predictions, references=references,
        lang="ko", model_type="klue/roberta-large", num_layers=17,
        device=device_name, batch_size=16,
    )
    f1_scores = [round(v * 100, 2) for v in results["f1"]]
    return round(float(np.mean(f1_scores)), 2), f1_scores


def calculate_nli_faithfulness(articles, predictions, batch_size=16):
    entail_scores = []
    for i in range(0, len(articles), batch_size):
        batch_arts = [art[:500] for art in articles[i: i + batch_size]]
        batch_preds = predictions[i: i + batch_size]

        # [핵심 수정] truncation(480)과 padding 목표 길이(512)를 분리.
        # 배치 전체가 패딩 없이 꽉 차면(padding_mask.all()==True) transformers의
        # SDPA 마스크 최적화 경로에서 device-side assert가 발생하는 버그가 있음.
        # 항상 최소 32개 패딩 토큰이 남도록 강제해서 이 조건 자체를 원천 차단.
        tokenized = nli_tokenizer(
            batch_arts, batch_preds, truncation=True, max_length=480,
            padding=False, return_token_type_ids=False,
        )
        inputs = nli_tokenizer.pad(
            tokenized, padding="max_length", max_length=512, return_tensors="pt",
        ).to(nli_model.device)

        with torch.no_grad():
            outputs = nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            entail_idx = 0
            for idx, label in nli_model.config.id2label.items():
                if "entail" in str(label).lower() or str(label) == "LABEL_0":
                    entail_idx = idx
                    break
            entail_scores.extend([round(float(s), 2) for s in (probs[:, entail_idx].cpu().numpy() * 100)])
    return round(float(np.mean(entail_scores)), 2), entail_scores


# ==========================================
# 3. 데이터 로드
# ==========================================
def load_domain_valid_data(domains, samples_per_domain):
    dfs = []
    for dom in domains:
        valid_csv = os.path.join(DATA_DIR, f"cleaned_dataset_valid_{dom}.csv")
        if not os.path.exists(valid_csv):
            raise FileNotFoundError(f"파일 없음: {valid_csv}")
        df_dom = pd.read_csv(valid_csv, encoding="utf-8-sig", low_memory=False)
        df_dom["article"] = df_dom["article"].fillna("").astype(str)
        df_dom["summary"] = df_dom["summary"].fillna("").astype(str)
        df_dom["domain"] = dom
        n_sample = min(samples_per_domain, len(df_dom))
        dfs.append(df_dom.sample(n=n_sample, random_state=42).reset_index(drop=True))
    return pd.concat(dfs, ignore_index=True)


# ==========================================
# 4. 생성 + EOS(캡) 판정
# ==========================================
def generate_and_check(model, tokenizer, texts, gen_kwargs, batch_size=8):
    summaries, hit_eos_flags = [], []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc="요약 생성 중"):
        batch_texts = texts[i: i + batch_size]
        inputs = tokenizer(
            batch_texts, max_length=512, truncation=True,
            padding=True, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
        for seq in output_ids:
            hit_eos_flags.append(tokenizer.eos_token_id in seq.tolist())
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        summaries.extend([s.strip() for s in decoded])
    return summaries, hit_eos_flags


# ==========================================
# 5. 메인: 모델별 순회 평가
# ==========================================
if __name__ == "__main__":
    df_eval_base = load_domain_valid_data(DOMAINS, EVAL_SAMPLES_PER_DOMAIN)
    print(f"검증 샘플: {len(df_eval_base):,}건 (모든 모델에 동일하게 사용)")

    articles = df_eval_base["article"].tolist()
    references = df_eval_base["summary"].tolist()

    summary_rows = []

    for train_per_domain, model_dir in MODEL_DIRS.items():
        if not os.path.exists(model_dir):
            print(f"모델 경로 없음 (스킵): {model_dir}")
            continue

        print("\n" + "#" * 90)
        print(f"# 평가 중: train_per_domain={train_per_domain:,} | {model_dir}")
        print("#" * 90)

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        if torch.cuda.is_available():
            model = model.to("cuda")

        df_eval = df_eval_base.copy()

        # [핵심 변경] 도메인별로 다른 generation 설정 적용 -> 도메인별로 나눠서 생성 후 병합
        pred_list, hit_eos_list = [None] * len(df_eval), [None] * len(df_eval)
        for dom in DOMAINS:
            dom_idx = df_eval.index[df_eval["domain"] == dom].tolist()
            dom_articles = df_eval.loc[dom_idx, "article"].tolist()
            dom_preds, dom_hit_eos = generate_and_check(
                model, tokenizer, dom_articles, GEN_KWARGS_BY_DOMAIN[dom], batch_size=8
            )
            for pos, i in enumerate(dom_idx):
                pred_list[i] = dom_preds[pos]
                hit_eos_list[i] = dom_hit_eos[pos]

        predictions = pred_list
        df_eval["pred_summary"] = predictions
        df_eval["hit_eos"] = hit_eos_list
        df_eval["gen_len"] = [len(tokenizer(p)["input_ids"]) for p in predictions]

        overall_rouge = calculate_rouge(predictions, references)
        overall_bertscore, bert_list = calculate_bertscore(predictions, references)
        overall_faith, entail_list = calculate_nli_faithfulness(articles, predictions)
        df_eval["bertscore_f1"] = bert_list
        df_eval["nli_faithfulness"] = entail_list

        overall_capped_ratio = (~df_eval["hit_eos"]).mean() * 100
        overall_gen_len_mean = df_eval["gen_len"].mean()

        row = {
            "train_per_domain": train_per_domain,
            "rouge1": overall_rouge["rouge1"],
            "rouge2": overall_rouge["rouge2"],
            "rougeL": overall_rouge["rougeL"],
            "bertscore": overall_bertscore,
            "nli_faithfulness": overall_faith,
            "gen_len_mean": round(overall_gen_len_mean, 2),
            "capped_ratio(%)": round(overall_capped_ratio, 1),
        }

        print(f"\n[{train_per_domain:,}] 전체 평균: ROUGE-1={overall_rouge['rouge1']} | "
              f"ROUGE-L={overall_rouge['rougeL']} | BERTScore={overall_bertscore} | "
              f"NLI={overall_faith} | gen_len={overall_gen_len_mean:.1f} | capped={overall_capped_ratio:.1f}%")

        for dom in DOMAINS:
            df_dom = df_eval[df_eval["domain"] == dom]
            d_rouge = calculate_rouge(df_dom["pred_summary"].tolist(), df_dom["summary"].tolist())
            d_gen_len = df_dom["gen_len"].mean()
            d_capped = (~df_dom["hit_eos"]).mean() * 100

            row[f"{dom}_rougeL"] = d_rouge["rougeL"]
            row[f"{dom}_gen_len_mean"] = round(d_gen_len, 2)
            row[f"{dom}_capped_ratio(%)"] = round(d_capped, 1)

            print(f"   * [{dom.upper()}] ROUGE-L={d_rouge['rougeL']} | gen_len={d_gen_len:.1f} | capped={d_capped:.1f}%")

        summary_rows.append(row)

        # 모델별 눈검수용 소량 샘플 저장
        df_qual = df_eval.groupby("domain").head(QUALITATIVE_SAVE_PER_DOMAIN).reset_index(drop=True)
        qual_path = os.path.join(DATA_DIR, f"qualitative_step_fixed_{train_per_domain}.csv")
        df_qual[["domain", "article", "summary", "pred_summary", "gen_len", "hit_eos",
                 "bertscore_f1", "nli_faithfulness"]].to_csv(qual_path, index=False, encoding="utf-8-sig")
        print(f"눈검수 샘플 저장: {qual_path}")

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 데이터량별 비교 표
    # ---------------------------------------------------------
    result_df = pd.DataFrame(summary_rows).sort_values("train_per_domain").reset_index(drop=True)

    print("\n" + "=" * 130)
    print("데이터량(train_per_domain)별 최종 비교 — 가설 B 판정용")
    print("=" * 130)
    print(result_df.to_string(index=False))
    print("=" * 130)

    result_df.to_csv(SUMMARY_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n비교 결과 저장 완료: {SUMMARY_OUTPUT_CSV}")
    print("\ngen_len_mean / capped_ratio(%)가 데이터량이 늘어도 거의 그대로면")
    print("   기존 90k 모델의 '길게 생성' 현상은 데이터량이 아니라 학습 스텝 수(오래 학습) 때문이었다는 게 확정됩니다.")
