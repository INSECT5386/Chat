from ddgs import DDGS
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def textrank_summarize(text, top_n=3):
    # 문장 분리
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= top_n:
        return text  # 문장 적으면 원본 리턴

    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # 코사인 유사도 행렬 생성
    sim_matrix = cosine_similarity(tfidf_matrix)

    # TextRank 점수 초기화
    scores = np.ones(len(sentences))
    d = 0.85  # damping factor

    for _ in range(20):  # 안정화 반복
        scores = (1 - d) + d * sim_matrix.dot(scores) / sim_matrix.sum(axis=1)

    # 점수 상위 문장 선택 및 원문 순서 정렬
    ranked = sorted(((score, idx) for idx, score in enumerate(scores)), reverse=True)
    selected_indices = sorted([idx for _, idx in ranked[:top_n]])

    # 요약문 생성
    summary = ' '.join([sentences[i] for i in selected_indices])
    return summary

def clean_summary_text(text):
    # 날짜 형식, 출처 문장 등 제거 (예: "March 28, 2024 - ", "DS Note" 등)
    text = re.sub(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4} - ', '', text)
    text = re.sub(r'\bDS Note\b', '', text)
    text = re.sub(r'\s{2,}', ' ', text)  # 연속 공백 하나로
    return text.strip()

# 그리고 요약 뒤에 clean_summary_text(summary) 적용


def search_web_and_summarize(query):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, region="wt-wt", safesearch="off", max_results=10):
            if "reddit.com" in r["href"]:
                continue  # Reddit은 스킵
            results.append({
                "title": r["title"],
                "href": r["href"],
                "snippet": r["body"]
            })

    # 제목+본문 붙여서 필터링 (짧거나 문장 미완성 스니펫은 걸러내기)
    combined_text_parts = []
    for item in results:
        title = item.get("title", "").strip()
        snippet = item.get("snippet", "").strip()

        if len(snippet) < 5:
            continue
        if snippet[-1] not in ".!?":
            continue

        combined_text_parts.append(f"{title}. {snippet}")

    combined_text = ' '.join(combined_text_parts)

    # 요약
    summary = textrank_summarize(combined_text, top_n=3)
    summary = clean_summary_text(summary)

    return summary, results

if __name__ == "__main__":
    query = input("검색어 입력 > ")
    summary, results = search_web_and_summarize(query)

    print("\n=== 요약 ===")
    print(summary)
    print("\n=== 검색 결과 참고 링크 ===")
    for item in results:
        print(f"{item['title']}\n{item['href']}\n")

    print("\n※ 위 내용은 인터넷 검색 결과를 바탕으로 요약한 것이며, 정확성을 100% 보장하지 않습니다.")
    print("※ 자세한 내용은 각 링크를 참고하세요.")
