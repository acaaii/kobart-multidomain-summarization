import os
import re
import json
import pandas as pd


# 1. 텍스트 정제(Cleaning) 함수
def clean_text(text, domain="general"):
    if not isinstance(text, str):
        return ""

    # 공통: 줄바꿈, 탭, 연속 공백을 하나의 공백으로 치환
    text = re.sub(r'\r\n|\r|\n|\t', ' ', text)

    # 뉴스(news) / 사설(editorial) 공통: 이메일 주소 제거
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

    if domain == "news":
        # "[서울=뉴시스]", "(서울=연합뉴스)", "홍길동 기자 =" 등 바이라인 표기 제거
        text = re.sub(r'\[.*?\]|[(<].*? 기자[)>]', '', text)
        text = re.sub(r'[가-힣]{2,4}\s*기자\s*[=:\-]', '', text)

    elif domain == "law":
        # 법률: 의미 없는 네모/세모 기호 등만 제한적 제거 (판례 번호, 법률 구두점은 유지)
        text = re.sub(r'[\■\▼\▶\★\⊙]', '', text)

    # 중복 공백 제거 후 앞뒤 공백 트리밍
    return re.sub(r'\s+', ' ', text).strip()


# 2. 개별 JSON 파일에서 article(원본) / summary(요약) 추출
def parse_and_clean_json(file_path, domain):
    print(f"파싱 시작: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = data.get('documents', [])
    parsed_records = []

    for doc in documents:
        # (1) 본문 문장 리스트들 하나로 병합
        sentences = []
        for paragraph in doc.get('text', []):
            for item in paragraph:
                if isinstance(item, dict) and 'sentence' in item:
                    sent = item['sentence']
                    # 이메일만 있는 문장 등은 배제
                    if '@' in sent and len(sent.split()) <= 3:
                        continue
                    sentences.append(sent)
                elif isinstance(item, str):
                    sentences.append(item)

        raw_article = " ".join(sentences)

        # (2) 요약문 병합
        abstractive = doc.get('abstractive', [])
        if isinstance(abstractive, list):
            raw_summary = " ".join(str(s) for s in abstractive)
        else:
            raw_summary = str(abstractive)

        # (3) 도메인별 정제 적용
        clean_article = clean_text(raw_article, domain)
        clean_summary = clean_text(raw_summary, domain)

        parsed_records.append({
            'id': doc.get('id', ''),
            'domain': domain,
            'article': clean_article,
            'summary': clean_summary,
            'article_len': len(clean_article),
            'summary_len': len(clean_summary)
        })

    print(f" └─> {len(parsed_records)}개 데이터 파싱 완료.")
    return pd.DataFrame(parsed_records)


# [설정] 파일 이름은 빼고, 파일이 담긴 폴더 경로만 지정 (앞에 r 필수!)
DATA_DIR = r"C:\Users\uze\PycharmProjects\sprint12"
DOMAINS = ["editorial", "law", "news"]
SPLIT = "valid"  # 'train', 'valid', 'test' 중 현재 불러올 분할 이름

all_dfs = []
for domain in DOMAINS:
    file_name = f"{SPLIT}_original_{domain}.json"
    file_path = os.path.join(DATA_DIR, file_name)

    if os.path.exists(file_path):
        df_domain = parse_and_clean_json(file_path, domain)
        all_dfs.append(df_domain)
    else:
        print(f"파일 없음 (스킵): {file_path}")

# 전체 도메인 통합
df_raw = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
print(f"\n초기 통합 데이터 개수: {len(df_raw)}개")

# -------------------------------------------------------------
# [이상치 필터링]
# 1. 본문(article)이 50자 미만으로 너무 짧거나
# 2. 요약문(summary)이 10자 미만인 불량/누락 데이터 제외
# -------------------------------------------------------------
df_clean = df_raw[
    (df_raw['article_len'] >= 50) &
    (df_raw['summary_len'] >= 10)
    ].drop_duplicates(subset=['article']).reset_index(drop=True)

print(f"필터링 및 중복 제거 후 최종 데이터 개수: {len(df_clean)}개")
print("\n[도메인별 최종 데이터 분포]")
print(df_clean['domain'].value_counts())


# 1. 정제 결과 랜덤 샘플 1개 눈으로 확인
sample = df_clean.sample(1).iloc[0]

print("=" * 60)
print(f"[도메인: {sample['domain']} | ID: {sample['id']}] 정제 결과 샘플")
print("-" * 60)
print("[정제된 본문]:\n", sample['article'][:300], "..." if len(sample['article']) > 300 else "")
print("-" * 60)
print("[정제된 요약문]:\n", sample['summary'])
print("=" * 60)

# 2. 학습 단계에서 편하게 불러올 수 있도록 최종 정제 데이터를 로컬 파일로 저장
output_file = f"cleaned_dataset_{SPLIT}.csv"
df_clean.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\n최종 정제된 데이터셋이 로컬 파일로 저장되었습니다: {output_file}")
