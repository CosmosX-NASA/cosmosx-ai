#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PMC NASA papers crawler (refactored)
- Reads a CSV of (title, link) pairs. The title/header may be in English ("title") or Korean ("제목"),
  but all internal variables and final outputs use the column name "title" only.
- For each link, fetches the HTML and extracts: title, first author, PMCID, abstract, PDF href, and date (YYYY.MM).
- Saves "info.csv" with columns [PMCID, title, date, lead_author, pdf_href, abstract].
- Saves "papers/{PMCID}.md" with a markdown summary extracted from the article page (#article-container).
"""

import os
import re
import csv
import time
import argparse
from typing import Dict, Optional, Tuple, List
from urllib.parse import urljoin

from tqdm import tqdm
import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36"
    )
}


# ---------------------- helpers ---------------------- #

def extract_year_month(ref_str: str) -> Dict[str, Optional[int]]:
    """Extract {year, month} from a citation-like string, e.g., "2021 Jan".
    Returns {} on failure.
    """
    try:
        # Some PMC citation strings look like: "Some Journal. 2021 Jan ..."
        parts = ref_str.split(".")
        candidate = parts[1] if len(parts) > 1 else ref_str
        match = re.search(r"(\d{4})\s+([A-Za-z]{3})", candidate)
        if not match:
            return {}
        year = int(match.group(1))
        month_str = match.group(2).capitalize()
        month_map = {
            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
            "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
        }
        month = month_map.get(month_str)
        if month is None:
            return {}
        return {"year": year, "month": month}
    except Exception:
        return {}


def count_total_rows(csv_path: str) -> int:
    total = 0
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        next(r, None)  # skip header
        for row in r:
            if not row or not any((cell or "").strip() for cell in row):
                continue
            total += 1
    return total


def _clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def _sanitize_for_csv(s: str) -> str:
    """Sanitize a field for CSV to avoid delimiter conflicts.
    - Replace commas with spaces.
    - Normalize whitespace.
    """
    return _clean_text((s or "").replace(",", " "))


def _safe_select_text(soup: BeautifulSoup, selector: str) -> str:
    el = soup.select_one(selector)
    return _clean_text(el.get_text()) if el else ""


# ---------------------- parsing ---------------------- #

def parse_article_html(html: str, base_url: str) -> Dict[str, str]:
    """Parse a single PMC article HTML and extract required fields + markdown summary."""
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one("#article-container") or soup

    # Title (prefer <h1>; fallback to meta)
    title_el = container.select_one("h1")
    title = _clean_text(title_el.get_text()) if title_el else _clean_text(
        (soup.find("meta", {"name": "citation_title"}) or {}).get("content")
    )

    # First author
    author_el = container.select_one(
        ".front-matter .name.western") or container.select_one(".name.western")
    first_author = _clean_text(author_el.get_text()) if author_el else ""

    # PMCID (prefer in-page; fallback to URL pattern)
    text_all = container.get_text(" ", strip=True)
    m = re.search(r"PMCID:\s*(PMC\d+)", text_all)
    pmcid = m.group(1) if m else None
    if not pmcid:
        m2 = re.search(r"/articles/(PMC\d+)/", base_url)
        pmcid = m2.group(1) if m2 else ""

    # Date (YYYY.MM) – be defensive
    citation_text = _safe_select_text(soup, ".pmc-layout__citation")
    date_info = extract_year_month(citation_text) if citation_text else {}
    date_str = f"{date_info.get('year', '')}.{date_info.get('month', '')}" if date_info else ""

    # Abstract
    abstract = ""
    abs_sec = container.select_one(
        "section.abstract") or container.select_one("#abstract1")
    if abs_sec:
        ps = abs_sec.find_all(["p", "div"])  # collect text blocks
        abstract = _clean_text("  \n\n".join(
            _clean_text(p.get_text(" ")) for p in ps))

    # PDF href
    pdf_href = None
    a = soup.select_one('a[data-ga-label="pdf_download_desktop"]')
    if not a:
        a = soup.select_one(
            'a[data-ga-label*="pdf_download"]') or soup.find("a", attrs={"title": "Download PDF"})
    if a and a.get("href"):
        pdf_href = urljoin(base_url, a["href"])
    else:
        meta_pdf = soup.find("meta", {"name": "citation_pdf_url"})
        pdf_href = meta_pdf.get(
            "content") if meta_pdf and meta_pdf.get("content") else ""

    # Authors (all)
    author_all = [_clean_text(e.get_text())
                  for e in container.select(".front-matter .name.western")]
    if not author_all and first_author:
        author_all = [first_author]

    # Markdown body (title, pmcid, authors, abstract, and article sections)
    md_lines: List[str] = []
    if title:
        md_lines.append(f"# {title}")
    if pmcid:
        md_lines.append(f"**PMCID**: {pmcid}")
    if author_all:
        md_lines.append(f"**Authors**: {', '.join(author_all)}")
    if abstract:
        md_lines.append("## Abstract")
        md_lines.append(abstract)

    body = container.select_one(
        ".body.main-article-body") or container.select_one(".main-article-body")
    if body:
        for node in body.find_all(["h2", "h3", "p"], recursive=True):
            if node.name in ("h2", "h3"):
                header_txt = _clean_text(node.get_text(" "))
                if header_txt:
                    md_lines.append(
                        ("## " if node.name == "h2" else "### ") + header_txt)
            elif node.name == "p":
                ptxt = _clean_text(node.get_text(" "))
                if ptxt:
                    md_lines.append(ptxt)

    md_content = "\n\n".join(md_lines).strip()

    return {
        "title": title or "",
        "date": date_str,
        "first_author": first_author or "",
        "pmcid": pmcid or "",
        "abstract": abstract or "",
        "pdf_href": pdf_href or "",
        "markdown": md_content,
    }


# ---------------------- CSV header detection ---------------------- #

def detect_columns_from_csv(csv_path: str) -> Tuple[str, str]:
    """Detect (title_col, link_col) from CSV header heuristically.
    Recognizes both English ("title") and Korean ("제목") headers, but maps to the variable name `title` internally.
    """
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, [])
        headers_lc = [h.lower() for h in headers]

    # candidates
    name_title_candidates = {"title", "paper", "article", "name", "제목"}
    name_link_candidates = {"link", "url", "href"}

    title_col = None
    link_col = None

    for i, h in enumerate(headers_lc):
        original = headers[i]
        if original in name_title_candidates or any(k in original.lower() for k in ["title", "제목"]):
            title_col = headers[i]
        if original in name_link_candidates or any(k in original.lower() for k in ["url", "link", "href"]):
            link_col = headers[i]

    if not link_col:
        # fallback: find any column with an http URL in the first data rows
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            dict_reader = csv.DictReader(f)
            for row in dict_reader:
                for h in headers:
                    val = (row.get(h) or "").strip()
                    if val.startswith("http"):
                        link_col = h
                        break
                if link_col:
                    break

    if not title_col:
        # default to the first non-link column, else first column
        for h in headers:
            if h != link_col:
                title_col = h
                break
        if not title_col and headers:
            title_col = headers[0]

    if not link_col and len(headers) >= 2:
        link_col = headers[1]

    if not title_col or not link_col:
        raise ValueError(
            f"Could not detect title/link columns from headers: {headers}")

    return title_col, link_col


# ---------------------- crawler ---------------------- #

def crawl(csv_path: str, outdir: str = ".", delay: float = 1.0, timeout: int = 30) -> None:
    papers_dir = os.path.join(outdir, "papers")
    os.makedirs(papers_dir, exist_ok=True)

    title_col, link_col = detect_columns_from_csv(csv_path)
    rows_info: List[Dict[str, str]] = []  # for info.csv

    # iterate rows
    total_rows = count_total_rows(csv_path)
    session = requests.Session()

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        dict_reader = csv.DictReader(f)
        for idx, row in enumerate(tqdm(dict_reader, total=total_rows), start=1):
            url = (row.get(link_col) or "").strip()
            title_hint = _clean_text(row.get(title_col) or "")
            if not url:
                continue

            try:
                resp = session.get(url, headers=HEADERS, timeout=timeout)
                resp.raise_for_status()
            except Exception as e:
                # minimal info on failure; keep columns consistent
                pmcid_guess = ""
                m = re.search(r"/articles/(PMC\d+)/", url)
                if m:
                    pmcid_guess = m.group(1)
                rows_info.append({
                    "PMCID": pmcid_guess,
                    "title": _sanitize_for_csv(title_hint),
                    "date": "",
                    "lead_author": "",
                    "pdf_href": "",
                    "abstract": f"Fetch error: {e}",
                })
                time.sleep(delay)
                continue

            parsed = parse_article_html(resp.text, url)

            # Write markdown
            pmcid = parsed.get("pmcid", "")
            md_filename = f"{pmcid}.md" if pmcid else f"NO_PMCID_{idx}.md"
            md_path = os.path.join(papers_dir, md_filename)
            try:
                with open(md_path, "w", encoding="utf-8") as mf:
                    mf.write(parsed.get("markdown", ""))
            except Exception:
                os.makedirs(papers_dir, exist_ok=True)
                with open(md_path, "w", encoding="utf-8") as mf:
                    mf.write(parsed.get("markdown", ""))

            # Accumulate info for info.csv
            output_title = parsed.get("title") or title_hint
            rows_info.append({
                "PMCID": pmcid,
                # avoid slashes in CSV cells
                "title": _sanitize_for_csv(output_title).replace("/", " "),
                "date": parsed.get("date", ""),
                "lead_author": _sanitize_for_csv(parsed.get("first_author", "")),
                "pdf_href": (parsed.get("pdf_href", "") or "").replace(",", ""),
                "abstract": _sanitize_for_csv(parsed.get("abstract", "")).replace("/", " "),
            })

            time.sleep(delay)

    # Save info.csv (ordered columns)
    info_path = os.path.join(outdir, "info.csv")
    fieldnames = ["PMCID", "title", "date",
                  "lead_author", "pdf_href", "abstract"]
    with open(info_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_info:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    # Post-process info.csv and prune orphaned markdown files
    df = pd.read_csv(info_path)
    # Keep only rows with PMCID and title (others may legitimately be empty)
    df = df.drop_duplicates().dropna(
        subset=["PMCID", "title"]).reset_index(drop=True)
    # Normalize titles (no slashes)
    df["title"] = df["title"].astype(str).str.replace("/", " ", regex=False)
    df.to_csv(info_path, index=False)

    # Remove markdown files that do not have a matching PMCID row
    keep_ids = set(df["PMCID"].astype(str).tolist())
    for fname in os.listdir(papers_dir):
        if not fname.endswith(".md"):
            continue
        base = os.path.splitext(fname)[0]
        if base not in keep_ids and not base.startswith("NO_PMCID_"):
            try:
                os.remove(os.path.join(papers_dir, fname))
            except FileNotFoundError:
                pass

    print(f"Saved: {info_path}")
    print(f"Papers dir: {papers_dir}")


# ---------------------- CLI ---------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Crawl PMC article pages and extract metadata + markdown summaries (title-only output)."
    )
    parser.add_argument("--csv", default="SB_publication_PMC_sample.csv",
                        help="Path to CSV file containing at least (title, link) columns.")
    parser.add_argument("--outdir", default=".",
                        help="Output directory (default: current directory).")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay seconds between requests (default: 1.0).")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Request timeout seconds (default: 30).")
    args = parser.parse_args()

    crawl(csv_path=args.csv, outdir=args.outdir,
          delay=args.delay, timeout=args.timeout)


if __name__ == "__main__":
    main()
