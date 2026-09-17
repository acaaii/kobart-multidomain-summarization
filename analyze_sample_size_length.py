"""
가설 A 최종 확인: 학습에 쓰는 것과 동일한 shuffle(seed=42).select(range(N)) 방식으로
N(5000/10000/20000/30000/전체)별 도메인 요약 길이 평균이 실제로 달라지는지 확인.

- 전체 모집단 분포는 이미 확인했지만(law가 구조적으로 다름), 혹시 "샘플링 자체가
  N에 따라 편향된 부분집합을 뽑는다"는 가능성까지 배제하기 위한 최종 sanity check.
- law는 tokenized_dataset_law_target190, editorial/news는 기존 tokenized_dataset_{domain} 사용
  (학습 스크립트와 동일한 경로 매핑)
"""

import os
import pandas as pd
from datasets import load_from_disk

DATA_DIR = r"C:\Users\uze\PycharmProjects\sprint12"

DOMAIN_DATASET_DIR = {
    "news": os.path.join(DATA_DIR, "tokenized_dataset_news"),
    "editorial": os.path.join(DATA_DIR, "tokenized_dataset_editorial"),
    "law": os.path.join(DATA_DIR, "tokenized_dataset_law_target190"),
}
DOMAINS = list(DOMAIN_DATASET_DIR.keys())

SAMPLE_SIZES = [5000, 10000, 20000, 30000, None]  # None = 전체
OUTPUT_CSV = os.path.join(DATA_DIR, "sample_size_length_check.csv")


def label_len(example):
    example["label_len"] = sum(1 for l in example["labels"] if l != -100)
    return example


def main():
    domain_train = {}
    for dom in DOMAINS:
        dom_dir = DOMAIN_DATASET_DIR[dom]
        if not os.path.exists(dom_dir):
            print(f"경로 없음 (스킵): {dom_dir}")
            continue
        ds = load_from_disk(dom_dir)["train"]
        print(f"[{dom}] label_len 계산 중... ({len(ds):,}건)")
        ds = ds.map(label_len, num_proc=1)
        domain_train[dom] = ds

    rows = []
    for n in SAMPLE_SIZES:
        n_label = "전체" if n is None else str(n)
        merged_lens = []
        row = {"N": n_label}

        for dom, ds in domain_train.items():
            size = len(ds) if n is None else min(n, len(ds))
            sampled = ds.shuffle(seed=42).select(range(size))
            lens = sampled["label_len"]
            mean_len = sum(lens) / len(lens)
            row[f"{dom}_mean"] = round(mean_len, 2)
            row[f"{dom}_count"] = size
            merged_lens.extend(lens)

        # 1:1:1 병합 시 전체 평균 (학습에 실제로 들어가는 병합 데이터 기준)
        row["merged_mean"] = round(sum(merged_lens) / len(merged_lens), 2)
        row["merged_count"] = len(merged_lens)
        rows.append(row)

    result_df = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print("N(샘플 수)별 도메인별 / 병합 요약 라벨 길이 평균 변화")
    print("=" * 100)
    print(result_df.to_string(index=False))
    print("=" * 100)

    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUTPUT_CSV}")

    # merged_mean이 N에 따라 얼마나 변하는지 한눈에 보이도록 변화폭 출력
    max_mean = result_df["merged_mean"].max()
    min_mean = result_df["merged_mean"].min()
    print(f"\nmerged_mean 변동폭: {min_mean} ~ {max_mean} (차이 {round(max_mean - min_mean, 2)} 토큰)")


if __name__ == "__main__":
    main()
