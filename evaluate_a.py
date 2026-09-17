import os
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
# 1. [설정] 환경 및 경로 설정
# ==========================================
DATA_DIR = r"C:\Users\uze\PycharmProjects\sprint12"

# 학습 완료된 요약 모델 경로
MODEL_DIR = os.path.join(DATA_DIR, "final_kobart_summary_model_90k")

# 평가할 도메인 목록
DOMAINS = ["news", "editorial", "law"]

# 정량 평가(ROUGE/BERTScore/NLI)를 위해 도메인당 추출할 샘플 수
EVAL_SAMPLES_PER_DOMAIN = 300

# [핵심 요청 1] 터미널 콘솔에 직접 출력하여 눈검수할 도메인당 샘플 수
CONSOLE_DISPLAY_PER_DOMAIN = 5

# [핵심 요청 2] 엑셀 파일로 저장할 도메인당 샘플 수 (총 60건)
QUALITATIVE_SAVE_PER_DOMAIN = 20

device = 0 if torch.cuda.is_available() else -1
device_name = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 Device: {device_name}")

# ==========================================
# 2. 평가 지표 및 보조 검증 모델 로드
# ==========================================
print("평가 지표 및 보조 검증 모델 로드 중...")

# 1) ROUGE (Mecab 또는 Kiwi 형태소 분석기 자동 감지)
rouge_metric = evaluate.load("rouge")
tokenizer_mode = "whitespace"

try:
    from mecab import MeCab

    mecab = MeCab()
    tokenizer_mode = "mecab"
    print(" [ROUGE] Mecab 형태소 분석기 적용 완료")
except ImportError:
    try:
        from kiwipiepy import Kiwi

        kiwi = Kiwi()
        tokenizer_mode = "kiwi"
        print(" [ROUGE] Kiwi 형태소 분석기 적용 완료 (Mecab 대체)")
    except ImportError:
        print(" [ROUGE] 형태소 분석기 미설치로 기본 띄어쓰기 기준 채점")

# 2) BERTScore
bertscore_metric = evaluate.load("bertscore")

# 3) NLI Model (AutoTokenizer/AutoModel 직접 로드로 CUDA assert 에러 차단)
NLI_MODEL_NAME = "Huffon/klue-roberta-base-nli"
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
if torch.cuda.is_available():
    nli_model = nli_model.to("cuda")
nli_model.eval()

print(" [BERTScore & NLI] 의미 유사도/사실 일치도 지표 준비 완료")


# ==========================================
# 3. 평가 연산 핵심 함수들
# ==========================================
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
        predictions=preds_m,
        references=refs_m,
        rouge_types=["rouge1", "rouge2", "rougeL"]
    )
    return {k: round(v * 100, 2) for k, v in results.items()}


def calculate_bertscore(predictions, references):
    """num_layers=17 명시로 Large 모델 KeyError 차단"""
    results = bertscore_metric.compute(
        predictions=predictions,
        references=references,
        lang="ko",
        model_type="klue/roberta-large",
        num_layers=17,
        device=device_name,
        batch_size=16
    )
    f1_scores = [round(v * 100, 2) for v in results["f1"]]
    mean_f1 = round(np.mean(f1_scores), 2)
    return mean_f1, f1_scores


def calculate_nli_faithfulness(articles, predictions, batch_size=16):
    """return_token_type_ids=False 적용으로 RoBERTa CUDA 인덱스 에러 완벽 차단"""
    entail_scores = []

    for i in tqdm(range(0, len(articles), batch_size), desc="[NLI] 사실 일치도(Entailment) 검증 중"):
        batch_arts = [art[:500] for art in articles[i: i + batch_size]]
        batch_preds = predictions[i: i + batch_size]

        inputs = nli_tokenizer(
            batch_arts,
            batch_preds,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_token_type_ids=False
        ).to(nli_model.device)

        with torch.no_grad():
            outputs = nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

            entail_idx = 0
            for idx, label in nli_model.config.id2label.items():
                if "entail" in str(label).lower() or str(label) == "LABEL_0":
                    entail_idx = idx
                    break

            batch_entail_scores = probs[:, entail_idx].cpu().numpy() * 100
            entail_scores.extend([round(float(s), 2) for s in batch_entail_scores])

    faithfulness_score = round(float(np.mean(entail_scores)), 2)
    return faithfulness_score, entail_scores


# ==========================================
# 4. 요약문 생성 함수 (Beam Search)
# ==========================================
def generate_summaries_batch(model, tokenizer, texts, batch_size=8):
    summaries = []
    model.eval()

    for i in tqdm(range(0, len(texts), batch_size), desc="요약문 생성 중"):
        batch_texts = texts[i: i + batch_size]
        inputs = tokenizer(
            batch_texts,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=128,
                min_length=15,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True
            )

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        summaries.extend([s.strip() for s in decoded])

    return summaries


# ==========================================
# 5. 도메인별 검증 CSV 로드 함수
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
# 6. 메인 평가 실행부
# ==========================================
if __name__ == "__main__":
    # 1) 요약 모델 로드
    print(f"\n요약 모델 로드 중: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    if torch.cuda.is_available():
        model = model.to("cuda")

    # 2) 검증 데이터 준비 (도메인당 300건씩 -> 총 900건)
    df_eval = load_domain_valid_data(DOMAINS, EVAL_SAMPLES_PER_DOMAIN)
    print(f"총 {len(df_eval):,}개 검증 샘플에 대한 복합 평가를 시작합니다.")

    articles = df_eval["article"].tolist()
    references = df_eval["summary"].tolist()

    # 3) AI 요약문 생성
    predictions = generate_summaries_batch(model, tokenizer, articles, batch_size=8)
    df_eval["pred_summary"] = predictions

    # ---------------------------------------------------------
    # 7. [정량 평가] ROUGE + BERTScore + NLI Faithfulness
    # ---------------------------------------------------------
    print("\n" + "=" * 85)
    print("[고급 통합 정량 평가 결과: 단어 일치도 + 의미 유사도 + 사실 부합도]")
    print("=" * 85)

    overall_rouge = calculate_rouge(predictions, references)
    overall_bertscore, bert_list = calculate_bertscore(predictions, references)
    overall_faithfulness, entail_list = calculate_nli_faithfulness(articles, predictions)

    df_eval["bertscore_f1"] = bert_list
    df_eval["nli_faithfulness"] = entail_list

    print(f" [전체 평균 ({len(df_eval):,}건)]")
    print(f"   ├─ ROUGE-1   : {overall_rouge['rouge1']}점 (키워드 포착)")
    print(f"   ├─ ROUGE-2   : {overall_rouge['rouge2']}점 (구절 유창성)")
    print(f"   ├─ ROUGE-L   : {overall_rouge['rougeL']}점 (문장 어순/구조)")
    print(f"   ├─ BERTScore : {overall_bertscore}점 (문맥 의미 일치도)")
    print(f"   └─ NLI Fact  : {overall_faithfulness}점 (원문 사실 부합도/Hallucination 방어율)\n")

    print(" [도메인별 세부 정량 비교]")
    for dom in DOMAINS:
        df_dom = df_eval[df_eval["domain"] == dom]
        if not df_dom.empty:
            d_preds = df_dom["pred_summary"].tolist()
            d_refs = df_dom["summary"].tolist()

            d_rouge = calculate_rouge(d_preds, d_refs)
            d_bert = round(df_dom["bertscore_f1"].mean(), 2)
            d_faith = round(df_dom["nli_faithfulness"].mean(), 2)

            print(
                f"   * [{dom.upper()}] R1={d_rouge['rouge1']} | R2={d_rouge['rouge2']} | RL={d_rouge['rougeL']} | BERTScore={d_bert} | NLI Fact={d_faith}"
            )
    print("=" * 85)

    # ---------------------------------------------------------
    # 8. [핵심 추가] 정성 평가: 도메인마다 5개씩 원문·정답·예측 눈검수 콘솔 출력
    # ---------------------------------------------------------
    print("\n" + "=" * 85)
    print("[정성 평가: 도메인별 원문 vs 정답 vs 예측 요약 5선 비교]")
    print("=" * 85)

    for dom in DOMAINS:
        df_dom_samples = df_eval[df_eval["domain"] == dom].head(CONSOLE_DISPLAY_PER_DOMAIN)
        print(f"\n>>>>>>>>> [{dom.upper()} 도메인 생성 품질 눈검수 (상위 {CONSOLE_DISPLAY_PER_DOMAIN}개)] <<<<<<<<<")

        for idx, (_, row) in enumerate(df_dom_samples.iterrows(), 1):
            print("-" * 85)
            # 콘솔 창 스크롤 가독성을 위해 원문은 앞 200자만 미리보기 표기
            article_preview = row["article"][:200] + ("..." if len(row["article"]) > 200 else "")
            print(f"[{dom.upper()} #{idx}]")
            print(f"[원본 문서]:\n  {article_preview}\n")
            print(f"[정답 요약 (Reference)]:\n  {row['summary']}\n")
            print(f"[AI 생성 요약 (Prediction)]:\n  {row['pred_summary']}\n")
            print(f"[평가 지표] BERTScore: {row['bertscore_f1']}점 | NLI 사실 부합도: {row['nli_faithfulness']}%")
        print("-" * 85)

    # ---------------------------------------------------------
    # 9. [엑셀 저장] 눈검수용 파일 저장: 도메인별 20개씩(총 60개) 추출하여 저장
    # ---------------------------------------------------------
    df_qualitative = (
        df_eval.groupby("domain")
        .head(QUALITATIVE_SAVE_PER_DOMAIN)
        .reset_index(drop=True)
    )

    output_csv = "evaluation_320k_advanced_results_samples.csv"
    save_cols = [
        "domain",
        "article",
        "summary",
        "pred_summary",
        "bertscore_f1",
        "nli_faithfulness",
    ]
    df_qualitative[save_cols].to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 85)
    print(f"도메인별 {QUALITATIVE_SAVE_PER_DOMAIN}건씩(총 {len(df_qualitative)}건) 눈검수 파일 저장 완료 -> {output_csv}")
    print("터미널에서 도메인별 5개씩 즉시 눈검수하시고, 상세 분석은 60개 샘플 엑셀 파일로 편하게 진행해 보세요!")
    print("=" * 85)
