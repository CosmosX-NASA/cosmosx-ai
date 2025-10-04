from transformers import TextClassificationPipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
from tqdm import tqdm
import pandas as pd
import os

# 문장이 너무 크면 잘 못받는다. 그런데 그 수치가 얼마인지 정확하게 모르겠음.(1500~3300)
model = AutoModelForSequenceClassification.from_pretrained(
    "pritamdeka/PubMedBert-PubMed200kRCT")
tokenizer = AutoTokenizer.from_pretrained(
    "pritamdeka/PubMedBert-PubMed200kRCT")
pipe = TextClassificationPipeline(
    model=model, tokenizer=tokenizer, return_all_scores=True)


def get_paper_paragraphs(paper_path: str) -> list[str]:
    if not os.path.exists(paper_path):
        raise Exception("Paper not found")

    with open(paper_path, 'r') as f:
        paper = f.read()

    return [paragraph for paragraph in paper.split('\n') if paragraph != '' and len(paragraph) > 150 and not paragraph.startswith('# ')]


def chunk_text(text: str, max_len: int = 1500, overlap: int = 500) -> List[str]:
    """
    주어진 문자열을 길이 max_len으로 잘라 겹치게(overlap) 분할한다.
    마지막 조각은 max_len 이하가 될 수 있으며, 항상 원문 순서를 유지한다.
    """
    if max_len <= 0:
        raise ValueError("max_len must be > 0")
    if not (0 <= overlap < max_len):
        raise ValueError("overlap must satisfy 0 <= overlap < max_len")

    n = len(text)
    if n <= max_len:
        return [text]

    step = max_len - overlap
    chunks = []
    for start in range(0, n, step):
        chunk = text[start:start + max_len]
        if not chunk:
            break
        chunks.append(chunk if '.' not in chunk else chunk.rsplit('.', 1)[0])
        if start + max_len >= n:
            break  # 끝까지 도달하면 종료
    return chunks


def split_lines_with_overlap(lines: List[str], max_len: int = 1500, overlap: int = 500) -> List[str]:
    """
    lines의 각 항목을 chunk_text 규칙으로 분할해 하나의 리스트로 이어 붙인다.
    원래 항목 순서가 유지되며, 각 항목 내부에서도 앞→뒤 순서가 유지된다.
    """
    result: List[str] = []
    for s in lines:
        result.extend(chunk_text(s, max_len=max_len, overlap=overlap))
    return result


# 논문 가져와서 문단 단위 split
def get_fit_paragraph(paper_name: str) -> str:
    paper_path = os.path.join(os.getcwd(), 'papers', paper_name)
    print(f'paper_path: {paper_path}')
    raw_paragraphs = get_paper_paragraphs(paper_path)
    paragraphs = split_lines_with_overlap(raw_paragraphs)

    # 초기화
    best_paragraph = {}
    labels = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']
    for label in labels:
        best_paragraph[label] = {
            'index': 0,
            'score': 0
        }

    for i in tqdm(range(len(paragraphs))):
        # print(f"paragraph {i}: length {len(paragraphs[i])}")
        inspect = pipe(paragraphs[i])[0]
        for item in inspect:
            if best_paragraph[item['label']]['score'] < item['score']:
                best_paragraph[item['label']]['score'] = item['score']
                best_paragraph[item['label']]['index'] = i

    final_report = ''

    for label in best_paragraph:
        final_report += f"## {label}: {best_paragraph[label]['score']}\n"
        final_report += paragraphs[best_paragraph[label]['index']]
        final_report += '\n\n'

    return final_report

# Microgravity will dilate human blood vessels and promote growth.


paper_list = [
    'PMC11988870',
    'PMC4110898',
    'PMC5515531',
    'PMC7787258',
    'PMC5460135',
]
os.makedirs('paragraph_results', exist_ok=True)
info_df = pd.read_csv('info.csv').loc[:, ['PMCID', 'title', 'abstract']]
for paper in paper_list:
    try:
        target_df = info_df[info_df['PMCID'] == 'PMC3630201']
        title = target_df['title'].values[0].replace('/', ',')
        abstract = target_df['abstract'].values[0].replace('/', ',')

        report = get_fit_paragraph(f'{paper}.md')
    except:
        print(f"Error: {paper}")
    with open(f'paragraph_results/{paper}.md', 'w') as f:
        f.write(f"# {title}\n\n")
        f.write(f"## Abstract\n{abstract}\n\n")
        f.write(report)

# """
#   [
#     { "label": "BACKGROUND", "score": 0.3304738402366638 },
#     { "label": "CONCLUSIONS", "score": 0.6330535411834717 },
#     { "label": "METHODS", "score": 0.026339353993535042 },
#     { "label": "OBJECTIVE", "score": 0.00555841252207756 },
#     { "label": "RESULTS", "score": 0.004574866499751806 }
#   ],
# """
