"""
truncation 없는 실제 원본 길이 분포 확인.

- tokenized_dataset은 이미 768/128로 잘린 상태라 "원래 얼마나 길었는지"는 알 수 없음.
- 그래서 원본 CSV(cleaned_dataset_train_{domain}.csv)를 truncation=False로 토큰화해서
  article(input)과 summary(label)의 실제 전체 길이 분포를 잰다.
- 도메인별 히스토그램 + 전체 도메인 합친 히스토그램을 각각 article/summary 두 세트로 그림.

대상 경로: DATA_DIR/cleaned_dataset_train_{domain}.csv  (domain: editorial, law, news)
필요 설치: pip install transformers pandas matplotlib
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from transformers import AutoTokenizer

# ==========================================
# 설정
# ==========================================
DATA_DIR = r"C:\Users\uze\PycharmProjects\sprint12"
MODEL_NAME = "gogamza/kobart-base-v2"
DOMAINS = ["editorial", "law", "news"]
SPLIT = "train"

CURRENT_MAX_INPUT_LENGTH = 768   # 참고선 (article truncation 기준)
CURRENT_MAX_TARGET_LENGTH = 128  # 참고선 (summary truncation 기준)

OUTPUT_STATS_CSV = os.path.join(DATA_DIR, "raw_length_full_stats.csv")
OUTPUT_ARTICLE_HIST_IMG = os.path.join(DATA_DIR, "article_len_full_histogram.png")
OUTPUT_SUMMARY_HIST_IMG = os.path.join(DATA_DIR, "summary_len_full_histogram.png")

try:
    rc("font", family="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    print("한글 폰트 설정 실패 — 그래프의 한글이 깨질 수 있습니다.")

COLORS = {"editorial": "#4C72B0", "law": "#C44E52", "news": "#55A868"}

print(f"토크나이저 로드 중: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ==========================================
# 원본 길이(truncation 없이) 계산
# ==========================================
def token_len(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    # truncation=False: 실제 전체 토큰 수를 그대로 잰다
    return len(tokenizer(text, truncation=False)["input_ids"])


def load_domain_lengths(domain: str, split: str):
    file_path = os.path.join(DATA_DIR, f"cleaned_dataset_{split}_{domain}.csv")
    if not os.path.exists(file_path):
        print(f"파일 없음 (스킵): {file_path}")
        return None

    df = pd.read_csv(file_path, encoding="utf-8-sig", low_memory=False)
    df["article"] = df["article"].fillna("").astype(str)
    df["summary"] = df["summary"].fillna("").astype(str)

    print(f"[{domain}/{split}] 원본 길이 계산 중 (truncation 없음)... ({len(df):,}건)")
    df["article_full_len"] = df["article"].apply(token_len)
    df["summary_full_len"] = df["summary"].apply(token_len)

    out = df[["article_full_len", "summary_full_len"]].copy()
    out["domain"] = domain
    return out


def plot_hist_with_combined(all_df, col, ref_line, out_path, xlabel, title_prefix):
    """도메인별 subplot 3개 + 전체 합친 subplot 1개, 총 2x2 구성"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, domain in enumerate(DOMAINS):
        ax = axes[i]
        sub = all_df[all_df["domain"] == domain][col]
        ax.hist(sub, bins=50, color=COLORS.get(domain), alpha=0.8)
        if ref_line:
            ax.axvline(ref_line, color="black", linestyle="--", linewidth=1.2)
        ax.set_title(f"{title_prefix} - {domain} (n={len(sub):,})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("문서 수")

    # 4번째 칸: 전체 도메인 합친 분포 (overlay)
    ax = axes[3]
    for domain in DOMAINS:
        sub = all_df[all_df["domain"] == domain][col]
        ax.hist(sub, bins=50, alpha=0.5, label=domain, color=COLORS.get(domain))
    if ref_line:
        ax.axvline(ref_line, color="black", linestyle="--", linewidth=1.2, label=f"현재 truncation 기준 ({ref_line})")
    ax.set_title(f"{title_prefix} - 전체 도메인 합친 분포")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("문서 수")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"저장 완료: {out_path}")


def main():
    frames = []
    for domain in DOMAINS:
        df = load_domain_lengths(domain, SPLIT)
        if df is not None:
            frames.append(df)

    if not frames:
        print("분석할 데이터가 없습니다. DATA_DIR / 파일명을 확인하세요.")
        return

    all_df = pd.concat(frames, ignore_index=True)

    # ---- 통계 테이블 (도메인별 + 전체) ----
    stats_by_domain = (
        all_df.groupby("domain")[["article_full_len", "summary_full_len"]]
        .describe(percentiles=[0.5, 0.95])
        .round(1)
    )
    stats_overall = (
        all_df[["article_full_len", "summary_full_len"]]
        .describe(percentiles=[0.5, 0.95])
        .round(1)
    )
    stats_overall.columns = pd.MultiIndex.from_product([["ALL"], stats_overall.columns])

    print("\n" + "=" * 100)
    print(f"원본(truncation 없음) article/summary 토큰 길이 통계 ({SPLIT} split)")
    print("=" * 100)
    print(stats_by_domain.to_string())
    print("-" * 100)
    print("[전체 도메인 합친 통계]")
    print(stats_overall.T.to_string())
    print("=" * 100)

    stats_by_domain.to_csv(OUTPUT_STATS_CSV, encoding="utf-8-sig")
    print(f"통계 CSV 저장 완료: {OUTPUT_STATS_CSV}")

    # 현재 truncation 기준 초과 비율 (진짜 잘리는 비율, 원본 기준이라 정확함)
    print(f"\n현재 MAX_INPUT_LENGTH({CURRENT_MAX_INPUT_LENGTH}) 실제 초과 비율:")
    for domain in DOMAINS:
        sub = all_df[all_df["domain"] == domain]["article_full_len"]
        if len(sub) == 0:
            continue
        over_ratio = (sub > CURRENT_MAX_INPUT_LENGTH).mean() * 100
        print(f"   - {domain}: {over_ratio:.1f}% 문서가 {CURRENT_MAX_INPUT_LENGTH} 토큰 초과 (truncation 발생)")

    print(f"\n현재 MAX_TARGET_LENGTH({CURRENT_MAX_TARGET_LENGTH}) 실제 초과 비율:")
    for domain in DOMAINS:
        sub = all_df[all_df["domain"] == domain]["summary_full_len"]
        if len(sub) == 0:
            continue
        over_ratio = (sub > CURRENT_MAX_TARGET_LENGTH).mean() * 100
        print(f"   - {domain}: {over_ratio:.1f}% 문서가 {CURRENT_MAX_TARGET_LENGTH} 토큰 초과 (truncation 발생)")

    # ---- 히스토그램: article ----
    plot_hist_with_combined(
        all_df, "article_full_len", CURRENT_MAX_INPUT_LENGTH,
        OUTPUT_ARTICLE_HIST_IMG, "article 전체 토큰 수", "Article 길이 분포"
    )

    # ---- 히스토그램: summary ----
    plot_hist_with_combined(
        all_df, "summary_full_len", CURRENT_MAX_TARGET_LENGTH,
        OUTPUT_SUMMARY_HIST_IMG, "summary 전체 토큰 수", "Summary 길이 분포"
    )


if __name__ == "__main__":
    main()
