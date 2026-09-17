"""
가설 C 검증: 재학습 없이, 기존 체크포인트의 generation 파라미터만 바꿔가며
length_penalty / num_beams / no_repeat_ngram_size 조합을 그리드서치.

- ROUGE + 평균 생성 길이(gen_len_mean)만 빠르게 비교 (BERTScore/NLI는 느려서 grid에서 제외)
- 특히 law 도메인만 따로 뽑아서 "law가 길게 생성되는 게 파라미터로 해결되는지" 확인
- 최적 조합을 찾은 뒤에는 기존 evaluate 스크립트(BERTScore/NLI 포함)에 그 값을 그대로 넣어
  최종 검증하는 걸 추천합니다.

기존 evaluate 스크립트와 동일한 로직(모델 로드, 데이터 로드, ROUGE 계산)을 재사용했습니다.
"""

import os
import itertools
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate

# ==========================================
# 1. 설정
# ==========================================
DATA_DIR = r"C:\Users\uze\PycharmProjects\sprint12"

# 그리드서치할 체크포인트 (원하는 모델 경로로 교체 가능)
MODEL_DIR = os.path.join(DATA_DIR, "final_kobart_summary_model_90k")

DOMAINS = ["news", "editorial", "law"]

# grid는 속도 위해 최종 평가(300건)보다 적게. 최적 조합 찾은 뒤 300건으로 재확인 권장.
GRID_EVAL_SAMPLES_PER_DOMAIN = 100

GEN_MAX_LENGTH = 128
GEN_MIN_LENGTH = 15

# [핵심] 그리드서치 파라미터 조합
LENGTH_PENALTIES = [0.6, 0.8, 1.0]
NUM_BEAMS_LIST = [2, 4]
NO_REPEAT_NGRAM_SIZES = [0, 3]

OUTPUT_CSV = os.path.join(DATA_DIR, "generation_grid_search_results.csv")

device = 0 if torch.cuda.is_available() else -1
device_name = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 Device: {device_name}")


# ==========================================
# 2. ROUGE 준비 (기존 스크립트와 동일 로직)
# ==========================================
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


# ==========================================
# 3. 데이터 로드 (기존 스크립트와 동일 로직)
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
        df_sampled = df_dom.sample(n=n_sample, random_state=42).reset_index(drop=True)
        dfs.append(df_sampled)
    return pd.concat(dfs, ignore_index=True)


# ==========================================
# 4. generation 파라미터를 인자로 받는 생성 함수
# ==========================================
def generate_summaries_batch(model, tokenizer, texts, gen_kwargs, batch_size=8):
    summaries = []
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        inputs = tokenizer(
            batch_texts, max_length=512, truncation=True,
            padding=True, return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        summaries.extend([s.strip() for s in decoded])

    return summaries


# ==========================================
# 5. 메인: 그리드서치
# ==========================================
if __name__ == "__main__":
    print(f"모델 로드 중: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    if torch.cuda.is_available():
        model = model.to("cuda")

    df_eval = load_domain_valid_data(DOMAINS, GRID_EVAL_SAMPLES_PER_DOMAIN)
    print(f"그리드서치용 검증 샘플: {len(df_eval):,}건")

    articles = df_eval["article"].tolist()
    references = df_eval["summary"].tolist()

    combos = list(itertools.product(LENGTH_PENALTIES, NUM_BEAMS_LIST, NO_REPEAT_NGRAM_SIZES))
    print(f"\n총 {len(combos)}개 조합 그리드서치 시작\n")

    results = []
    for length_penalty, num_beams, no_repeat_ngram_size in tqdm(combos, desc="그리드서치 진행"):
        gen_kwargs = dict(
            max_length=GEN_MAX_LENGTH,
            min_length=GEN_MIN_LENGTH,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=True,
        )
        if no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

        predictions = generate_summaries_batch(model, tokenizer, articles, gen_kwargs, batch_size=8)
        df_eval["_pred"] = predictions
        gen_lens = [len(tokenizer(p)["input_ids"]) for p in predictions]

        overall_rouge = calculate_rouge(predictions, references)
        row = {
            "length_penalty": length_penalty,
            "num_beams": num_beams,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "rouge1": overall_rouge["rouge1"],
            "rouge2": overall_rouge["rouge2"],
            "rougeL": overall_rouge["rougeL"],
            "gen_len_mean": round(sum(gen_lens) / len(gen_lens), 2),
        }

        # law 도메인만 별도로 gen_len 확인 (핵심 관찰 대상)
        for dom in DOMAINS:
            dom_mask = df_eval["domain"] == dom
            dom_preds = df_eval.loc[dom_mask, "_pred"].tolist()
            dom_lens = [len(tokenizer(p)["input_ids"]) for p in dom_preds]
            row[f"{dom}_gen_len_mean"] = round(sum(dom_lens) / len(dom_lens), 2)

        results.append(row)

    result_df = pd.DataFrame(results).sort_values("rougeL", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 130)
    print("Generation 파라미터 그리드서치 결과 (rougeL 기준 내림차순)")
    print("=" * 130)
    print(result_df.to_string(index=False))
    print("=" * 130)

    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장 완료: {OUTPUT_CSV}")
    print("\nrougeL이 비슷하다면 gen_len_mean(특히 law_gen_len_mean)이 낮은 쪽을 선택하는 걸 추천합니다.")
    print("   최적 조합을 고른 뒤, 그 값을 기존 evaluate 스크립트의 generate_summaries_batch generate() 인자에")
    print("   반영해서 BERTScore/NLI 포함 최종 검증(300건)을 진행하세요.")
