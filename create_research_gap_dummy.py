#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
research_gaps_extracted.parquet 과 research_id_pmc.csv 를 정확 매칭하여
SQLite3 에 넣을 INSERT 문들을 담은 SQL 파일을 생성합니다.

사용 예:
  python make_sql.py \
    --gaps research_gaps_extracted.parquet \
    --map research_id_pmc.csv \
    --out insert_research_gaps.sql
"""

import argparse
from pathlib import Path
import sys
import pandas as pd


def _escape_sql_str(s: str) -> str:
    """
    SQLite용 문자열 이스케이프: 작은따옴표(') -> 두 개('')
    None/NaN 은 빈 문자열로 처리.
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s)
    return s.replace("'", "''")


def generate_sql(gaps_parquet: Path, idmap_csv: Path, out_sql: Path) -> None:
    # -------- 입력 로드 및 유효성 검사 --------
    try:
        gaps = pd.read_parquet(gaps_parquet)
    except Exception as e:
        print(
            f"[오류] Parquet 파일을 읽는 중 문제 발생: {gaps_parquet}\n{e}", file=sys.stderr)
        sys.exit(1)

    try:
        idmap = pd.read_csv(
            idmap_csv, dtype={"id": "Int64", "pmc_id": "string", "title": "string"})
    except Exception as e:
        print(f"[오류] CSV 파일을 읽는 중 문제 발생: {idmap_csv}\n{e}", file=sys.stderr)
        sys.exit(1)

    required_gaps_cols = {"pmcid", "type", "content", "evidence"}
    required_idmap_cols = {"pmc_id", "id", "title"}

    missing_gaps = required_gaps_cols - set(gaps.columns)
    missing_idmap = required_idmap_cols - set(idmap.columns)

    if missing_gaps:
        print(
            f"[오류] Parquet에 필요한 열이 없습니다: {sorted(missing_gaps)}", file=sys.stderr)
        sys.exit(1)
    if missing_idmap:
        print(
            f"[오류] CSV에 필요한 열이 없습니다: {sorted(missing_idmap)}", file=sys.stderr)
        sys.exit(1)

    # 문자열/공백 정리(정확 매칭을 위해 공백만 제거)
    gaps["pmcid"] = gaps["pmcid"].astype("string").str.strip()
    idmap["pmc_id"] = idmap["pmc_id"].astype("string").str.strip()

    # 매칭 불가/결측 제거(정확 매칭 원칙 준수)
    gaps_valid = gaps.dropna(subset=["pmcid"])
    idmap_valid = idmap.dropna(subset=["pmc_id", "id", "title"])

    # -------- 정확 매칭 조인 --------
    # many_to_one: gaps의 여러 행이 같은 pmc_id 하나를 가리키는 건 허용,
    # idmap 쪽은 동일 pmc_id가 여러 번 나오면 오류를 내 정확성 보장.
    try:
        merged = gaps_valid.merge(
            idmap_valid[["pmc_id", "id", "title"]],
            left_on="pmcid",
            right_on="pmc_id",
            how="inner",
            validate="many_to_one",
        )
    except pd.errors.MergeError as e:
        print(
            "[오류] CSV에 같은 pmc_id가 여러 번 존재하여 '정확 매칭' 조건을 위반했습니다.\n"
            f"세부: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # 원본 gaps 순서를 최대한 보존(자동증가 id는 INSERT 순서대로 부여됨)
    merged = merged.reset_index().sort_values("index", kind="stable")

    matched_cnt = len(merged)
    total_gaps = len(gaps)
    dropped_cnt = total_gaps - matched_cnt

    if matched_cnt == 0:
        print("[경고] 정확 매칭된 행이 없습니다. SQL 파일을 생성하지 않습니다.", file=sys.stderr)
        sys.exit(2)

    # -------- SQL 생성 --------
    with open(out_sql, "w", encoding="utf-8") as f:
        f.write("-- 자동 생성된 SQLite3 INSERT 스크립트\n")
        f.write("BEGIN TRANSACTION;\n")

        # research_gaps INSERT: PK(id)는 자동 증가 → 열에서 제외
        for _, row in merged.iterrows():
            col_type = _escape_sql_str(row["type"])
            col_content = _escape_sql_str(row["content"])
            col_evidence = _escape_sql_str(row["evidence"])
            col_title = _escape_sql_str(row["title"])

            # research_id는 정수
            try:
                research_id = int(row["id"])
            except Exception:
                # Int64(Na) 등 예외 처리
                print(
                    f"[스킵] research_id가 정수가 아님: pmcid={row.get('pmcid')}, id={row.get('id')}",
                    file=sys.stderr,
                )
                continue

            sql = (
                'INSERT INTO "research_gaps" ("type", "content", "evidence", "research_title", "research_id") '
                f"VALUES ('{col_type}', '{col_content}', '{col_evidence}', '{col_title}', {research_id});\n"
            )
            f.write(sql)

        f.write("COMMIT;\n")

    # -------- 요약 출력 --------
    print(
        f"[완료] SQL 파일 생성: {out_sql}\n"
        f"- 총 gaps 행 수: {total_gaps}\n"
        f"- 정확 매칭된 행 수: {matched_cnt}\n"
        f"- 매칭 실패/제외된 행 수: {dropped_cnt}\n"
        f"- 참고: 매칭 기준은 pmcid(Parquet) == pmc_id(CSV) 의 공백 제거 후 '정확 일치'입니다."
    )


def main():
    parser = argparse.ArgumentParser(
        description="research_gaps용 SQLite INSERT SQL 생성기")
    parser.add_argument(
        "--gaps",
        type=Path,
        default=Path("research_gaps_extracted.parquet"),
        help="연구 간극 Parquet 파일 경로 (예: research_gaps_extracted.parquet)",
    )
    parser.add_argument(
        "--map",
        type=Path,
        default=Path("research_id_pmc.csv"),
        help="pmcid ↔ (id, title) 매핑 CSV 경로 (예: research_id_pmc.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("insert_research_gaps.sql"),
        help="생성될 SQL 파일 경로 (예: insert_research_gaps.sql)",
    )
    args = parser.parse_args()

    generate_sql(args.gaps, args.map, args.out)


if __name__ == "__main__":
    main()
