# Keyword extraction with KeyBERT + SciBERT (no scispaCy)
from transformers import pipeline
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text as sklearn_text
import pandas as pd
import os

# 1) SciBERT 임베딩 백엔드 준비 (GPU 사용 시 device=0)
hf_model = pipeline(
    "feature-extraction",
    model="allenai/scibert_scivocab_uncased",
    tokenizer="allenai/scibert_scivocab_uncased",
    device=-1
)

hf_model.tokenizer.model_max_length = 512

kw_model = KeyBERT(model=hf_model)


def extract_keywords(text: str, top_n: int = 20) -> list[str]:
    """
    바이오/과학 논문 텍스트에서 키워드(1~3그램)를 추출하여 리스트로 반환.
    """
    # 2) 도메인 불용어(섹션명 등) 보강
    domain_stop = {
        "et", "al", "figure", "fig", "supplementary",
        "introduction", "methods", "materials",
        "results", "discussion", "conclusion", "abstract"
    }
    # stop_words = sklearn_text.ENGLISH_STOP_WORDS.union(domain_stop)
    stop_words = list(sklearn_text.ENGLISH_STOP_WORDS.union(domain_stop))

    # 3) 후보 구(1~3-gram) 생성용 벡터라이저
    vectorizer = CountVectorizer(
        ngram_range=(1, 3),
        stop_words=stop_words,
        min_df=1
    )

    # 4) MMR로 중복 줄이며 Top-N 키프레이즈 선정
    keywords = kw_model.extract_keywords(
        text,
        vectorizer=vectorizer,
        use_mmr=True,           # 다양성-정보성 균형
        diversity=0.6,
        top_n=top_n
    )
    # [('keyword', score), ...] -> ['keyword', ...]
    return [kw for kw, score in keywords]


def get_keyword(row):
    kws = extract_keywords(row['abstract'], top_n=5)
    keywords_str = '.'.join(kws)

    row['keywords'] = keywords_str
    return row


if __name__ == "__main__":
    df = pd.read_csv('info.csv')
    df['keywords'] = ''
    df = df.apply(lambda row: get_keyword(row), axis=1)
    df = df.reset_index(drop=True)
    df.to_csv('keyword_info.csv', index=None)
