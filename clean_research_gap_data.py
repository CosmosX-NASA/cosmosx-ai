# -*- coding: utf-8 -*-
"""
유연한 규칙 기반 파서:
- 각 md 파일에서 "Conceptual / Methodological / Empirical" 섹션을 찾아 구간을 분리합니다.
- 섹션 내부에서 "Gap:", "Why it matters:", "Future research:" 라벨을 케이스/형태(볼드, 불릿 등) 불문하고 탐색합니다.
- 누락된 항목은 None 처리합니다.
- 결과는 (pmcid, type, content, evidence) 컬럼의 pandas.DataFrame으로 생성하고 CSV로 저장합니다.
"""
import re
from pathlib import Path
import pandas as pd
from typing import Dict, Optional, Tuple, List

DATA_DIR = Path('ResearchGap_DB')


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()

# 패턴 유틸: 라벨 블록 추출 (다음 라벨/섹션 시작 전까지 멀티라인 수집)


def extract_label_block(section_text: str, label: str) -> Optional[str]:
    # label은 'Gap' | 'Why it matters' | 'Future research'
    # 불릿(-,*,1.), 볼드(** **), 콜론 유무 등에 대응
    label_pattern = rf"(?:^|\n)\s*(?:[-*]|\d+\.)?\s*(?:\*\*)?\s*{label}\s*(?:\*\*)?\s*:?\s*"
    # 다음 라벨 또는 섹션 경계(###, **Conceptual..., 혹은 'Conceptual/Methodological/Empirical' 단어가 포함된 라인) 전까지 캡처
    next_label_lookahead = r"(?=(?:\n\s*(?:[-*]|\d+\.)?\s*(?:\*\*)?\s*(?:Gap|Why\s+it\s+matters|Future\s+research)(?:\s*\*\*)?\s*:)|" \
                           r"(?:\n\s*#{1,6}\s)|" \
                           r"(?:\n\s*\*\*?\s*(?:Conceptual|Methodological|Empirical)[^\n]*\*\*?)|" \
                           r"(?:\n\s*(?:Conceptual|Methodological|Empirical)\s+Gaps?\b)|$)"
    pattern = re.compile(label_pattern + r"(.*?)" +
                         next_label_lookahead, re.IGNORECASE | re.DOTALL)
    m = pattern.search(section_text)
    if not m:
        return None
    value = m.group(1).strip()
    # 불필요한 마크다운 기호 제거
    value = re.sub(r"^\s*[-*]\s*", "", value)  # 선행 불릿 제거
    value = re.sub(r"\s*\n\s*[-*]\s*", "\n- ", value)  # 내부 불릿은 유지
    value = value.strip() or None
    return value


def split_sections_by_type(text: str) -> Dict[str, str]:
    """
    파일 전체 텍스트에서 각 타입(Conceptual/Methodological/Empirical) 섹션을 찾아 구간 텍스트로 반환.
    다양한 헤딩 형태(#, **, 일반 텍스트 + Gaps 등)에 대응.
    """
    # 모든 줄 인덱스 확보
    lines = text.splitlines()
    indices: List[Tuple[str, int]] = []
    for i, line in enumerate(lines):
        ln = line.strip()
        # 섹션 키워드 감지 (헤딩/볼드/일반 라인 모두 허용)
        for t in ["Conceptual", "Methodological", "Empirical"]:
            if re.search(rf"\b{t}\b", ln, flags=re.IGNORECASE) and re.search(r"\bGap", ln, flags=re.IGNORECASE):
                indices.append((t.capitalize(), i))
                break
    # 인덱스가 전혀 없을 때 대비(파일 내에 명시가 없으면 전체를 대상으로 시도)
    if not indices:
        return {t: text for t in ["Conceptual", "Methodological", "Empirical"]}
    # 구간 자르기
    indices.sort(key=lambda x: x[1])
    sections: Dict[str, str] = {}
    for idx, (t, start) in enumerate(indices):
        end = indices[idx + 1][1] if idx + 1 < len(indices) else len(lines)
        section_text = "\n".join(lines[start:end])
        sections[t] = section_text
    return sections


def parse_file(path: Path) -> pd.DataFrame:
    raw = read_text(path)
    pmcid = path.stem  # 'PMC3166430'
    sections = split_sections_by_type(raw)

    rows = []
    for t in ["Conceptual", "Methodological", "Empirical"]:
        section_text = sections.get(t, "")
        gap = extract_label_block(section_text, "Gap")
        why = extract_label_block(section_text, "Why\s+it\s+matters")
        future = extract_label_block(section_text, "Future\s+research")
        # evidence 조합: 존재하는 것만 "\n\n"로 연결, 전부 None이면 None
        parts = [p for p in [why, future] if p]
        evidence = "\n\n".join(parts) if parts else None
        rows.append({"pmcid": pmcid, "type": t,
                    "content": gap, "evidence": evidence})
    return pd.DataFrame(rows, columns=["pmcid", "type", "content", "evidence"])


# 실제 실행: /mnt/data/PMC*.md 모두 처리
files = sorted(DATA_DIR.glob("PMC*.md"))
all_dfs = [parse_file(p) for p in files]
result_df = pd.concat(all_dfs, ignore_index=True)

# CSV 저장
out_path = "research_gaps_extracted.csv"
result_df.to_csv(out_path, index=False, encoding="utf-8")

out_path, result_df.shape
