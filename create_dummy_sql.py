#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data.parquet -> populate.sql 생성 스크립트

사용법:
  python make_sql.py --input data.parquet --output populate.sql
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


def sql_quote(value: str) -> str:
    """SQLite용 문자열 이스케이프 및 quoting."""
    if value is None:
        return "NULL"
    # 문자열로 강제 변환
    s = str(value)
    # 단일 인용부호 이스케이프
    s = s.replace("'", "''")
    return f"'{s}'"


def to_iso_date(value) -> str:
    """
    release_date 변환:
    - 'YYYY.MM' 또는 'YYYY.M' -> 'YYYY-MM-01'
    - 'YYYY-MM-DD' 등 일반 날짜 -> 해당 날짜
    - 'YYYY' -> 'YYYY-01-01'
    - 판독 불가 -> 'YYYY-01-01'(연도만 추출) 또는 None
    """
    if pd.isna(value):
        return None

    s = str(value).strip()

    # 1) YYYY.MM or YYYY.M
    m = re.fullmatch(r"^\s*(\d{4})\.(\d{1,2})\s*$", s)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        month = max(1, min(12, month))  # 범위 보정
        return f"{year:04d}-{month:02d}-01"

    # 2) YYYY-MM-DD, YYYY/MM/DD 등 일반 날짜 파싱
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # 3) pandas의 to_datetime 시도(강건성)
    try:
        dt = pd.to_datetime(s, errors="raise")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        pass

    # 4) YYYY만 주어진 경우
    m = re.fullmatch(r"^\s*(\d{4})\s*$", s)
    if m:
        year = int(m.group(1))
        return f"{year:04d}-01-01"

    # 5) float로 들어오는 경우(예: 2014.8 -> 2014-08-01 으로 **추정**)
    if isinstance(value, (int, float)):
        year = int(value)
        frac = float(value) - year
        # 소수점 첫째 자리만 유의미하다고 **추정** (0.1~0.12 -> 월)
        approx_month = round(frac * 10)
        month = approx_month if 1 <= approx_month <= 12 else 1
        return f"{year:04d}-{month:02d}-01"

    # 최종 실패 시 None
    return None


def parse_figures(figures_field):
    """
    figures_json은 문자열로 된 JSON 배열 또는 이미 리스트일 수 있음.
    각 요소는 {"url": "...", "caption": "..."} 형태.
    """
    if figures_field is None or (isinstance(figures_field, float) and pd.isna(figures_field)):
        return []

    if isinstance(figures_field, list):
        return figures_field

    s = str(figures_field).strip()
    if not s:
        return []

    try:
        data = json.loads(s)
        if isinstance(data, list):
            return data
        # 혹시 2중 인코딩된 문자열일 경우 한 번 더
        if isinstance(data, str):
            data2 = json.loads(data)
            if isinstance(data2, list):
                return data2
    except Exception:
        pass

    raise ValueError("figures_json 필드를 파싱할 수 없습니다.")


def main(input_path: str, output_path: str):
    # 데이터 로드
    df = pd.read_parquet(input_path)

    # 필드명 매핑 확인(결측 없음이 전제)
    required_cols = [
        "PMCID",
        "title",
        "date",
        "lead_author",
        "pdf_href",
        "abstract",          # 사용하지 않지만 존재 확인(설명과 일치 검증용)
        "figures_json",
        "keywords",          # 사용하지 않지만 존재 확인(설명과 일치 검증용)
        "overall_summary",
        "methods",
        "result",
        "brief_summary",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"입력 데이터에 다음 필드가 없습니다: {missing}")

    # SQL 파일 생성
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        f.write("-- Auto-generated SQL insert script\n")
        f.write("PRAGMA foreign_keys=ON;\n")
        f.write("BEGIN TRANSACTION;\n\n")

        # 1) research 테이블 INSERT
        f.write("-- ==========================\n")
        f.write("-- Insert into research table\n")
        f.write("-- ==========================\n")

        for _, row in df.iterrows():
            pmcid = row["PMCID"]
            title = row["title"]
            journal = "IJMS"  # 전건 동일
            doi = row["pdf_href"]
            author = row["lead_author"]
            release_date = to_iso_date(row["date"])
            brief_summary = row["brief_summary"]
            overall_summary = row["overall_summary"]
            methods = row["methods"]
            results = row["result"]

            cols = "(title, journal, doi, author, release_date, brief_summary, overall_summary, methods, results, pmc_id)"
            vals = (
                f"{sql_quote(title)}, "
                f"{sql_quote(journal)}, "
                f"{sql_quote(doi)}, "
                f"{sql_quote(author)}, "
                f"{sql_quote(release_date)}, "
                f"{sql_quote(brief_summary)}, "
                f"{sql_quote(overall_summary)}, "
                f"{sql_quote(methods)}, "
                f"{sql_quote(results)}, "
                f"{sql_quote(pmcid)}"
            )
            f.write(f"INSERT INTO research {cols} VALUES ({vals});\n")
        f.write("\n")

        # 2) figure 테이블 INSERT
        f.write("-- ========================\n")
        f.write("-- Insert into figure table\n")
        f.write("-- ========================\n")

        for _, row in df.iterrows():
            pmcid = row["PMCID"]
            figures = parse_figures(row["figures_json"])
            for fig in figures:
                url = fig.get("url")
                caption = fig.get("caption")
                # research_id는 pmc_id로 조회
                f.write(
                    "INSERT INTO figure (url, caption, research_id)\n"
                    f"SELECT {sql_quote(url)}, {sql_quote(caption)}, r.id FROM research r WHERE r.pmc_id = {sql_quote(pmcid)};\n"
                )

        f.write("\nCOMMIT;\n")

    print(f"완료: {out.resolve()} 에 SQL 스크립트를 생성했습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SQL insert script for research/figure tables from data.parquet")
    parser.add_argument("--input", "-i", default="data.parquet",
                        help="입력 Parquet 파일 경로 (기본: data.parquet)")
    parser.add_argument("--output", "-o", default="populate.sql",
                        help="출력 SQL 파일 경로 (기본: populate.sql)")
    args = parser.parse_args()
    main(args.input, args.output)
