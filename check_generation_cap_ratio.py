"""
가설 확인: law 요약이 length_penalty와 무관하게 길이가 똑같다는 건
-> max_length(128)에서 EOS 없이 강제로 잘리고 있다는 뜻일 가능성이 높음.
이걸 직접 확인: 생성된 output_ids 안에 eos_token_id가 있는지 검사.
- eos_token_id가 있으면 -> 모델이 스스로 문장을 끝냄 (자연 종료)
- eos_token_id가 없으면 -> max_length까지 다 채우고 강제 종료됨 (캡에 걸림)

기존 evaluate 스크립트와 동일한 모델/데이터 로드 로직 재사용.
"""

import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==========================================
# 설정
# ==========================================
DATA_DIR = r"C:\Users\uze\PycharmProjects\sprint12"
MODEL_DIR = os.path.join(DATA_DIR, "final_kobart_summary_model_15k")
DOMAINS = ["news", "editorial", "law"]

EVAL_SAMPLES_PER_DOMAIN = 300  # 캡 비율은 노이즈에 민감하니 넉넉하게

# 그리드서치에서 채택한 조합
GEN_KWARGS = dict(
    max_length=128,
    min_length=15,
    num_beams=4,
    length_penalty=0.8,
    no_repeat_ngram_size=3,
    early_stopping=True,
)

OUTPUT_CSV = os.path.join(DATA_DIR, "generation_cap_ratio_check.csv")

device_name = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 Device: {device_name}")


# ==========================================
# 데이터 로드 (기존 스크립트와 동일)
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
# 생성 + EOS 포함 여부 판정
# ==========================================
def generate_and_check_eos(model, tokenizer, texts, gen_kwargs, batch_size=8):
    """각 생성 결과가 EOS로 자연 종료됐는지(True) / max_length까지 강제로 채워졌는지(False) 반환"""
    hit_eos_flags = []
    raw_lens = []
    model.eval()

    for i in tqdm(range(0, len(texts), batch_size), desc="요약 생성 + EOS 검사 중"):
        batch_texts = texts[i: i + batch_size]
        inputs = tokenizer(
            batch_texts, max_length=512, truncation=True,
            padding=True, return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        for seq in output_ids:
            seq_list = seq.tolist()
            has_eos = tokenizer.eos_token_id in seq_list
            hit_eos_flags.append(has_eos)
            # pad/eos 제외한 실제 생성 길이 (special token 제외 디코드 후 재토큰화로 근사)
            raw_lens.append(len(seq_list))

    return hit_eos_flags, raw_lens


# ==========================================
# 메인
# ==========================================
if __name__ == "__main__":
    print(f"모델 로드 중: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    if torch.cuda.is_available():
        model = model.to("cuda")

    df_eval = load_domain_valid_data(DOMAINS, EVAL_SAMPLES_PER_DOMAIN)
    print(f"검증 샘플: {len(df_eval):,}건 (조합: {GEN_KWARGS})")

    articles = df_eval["article"].tolist()
    hit_eos_flags, raw_lens = generate_and_check_eos(model, tokenizer, articles, GEN_KWARGS, batch_size=8)

    df_eval["hit_eos"] = hit_eos_flags   # True = 자연 종료, False = 강제 캡
    df_eval["raw_len"] = raw_lens

    print("\n" + "=" * 90)
    print(f"도메인별 EOS 자연 종료 vs max_length({GEN_KWARGS['max_length']}) 강제 캡 비율")
    print("=" * 90)

    rows = []
    for dom in DOMAINS:
        df_dom = df_eval[df_eval["domain"] == dom]
        n = len(df_dom)
        capped_ratio = (~df_dom["hit_eos"]).mean() * 100
        avg_len = df_dom["raw_len"].mean()
        rows.append({
            "domain": dom,
            "count": n,
            "capped_ratio(%)": round(capped_ratio, 1),
            "natural_stop_ratio(%)": round(100 - capped_ratio, 1),
            "avg_raw_len": round(avg_len, 1),
        })
        print(f"   * [{dom.upper()}] 강제 캡: {capped_ratio:.1f}% | 자연 종료: {100 - capped_ratio:.1f}% | 평균 길이: {avg_len:.1f}")

    print("=" * 90)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장 완료: {OUTPUT_CSV}")
    print("\ncapped_ratio가 law만 유독 높다면, length_penalty가 안 먹히는 이유가")
    print("   '애초에 max_length까지 강제로 채워지고 있기 때문'이라는 게 확정됩니다.")
