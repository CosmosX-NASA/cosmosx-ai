#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified PMC crawler with figures
- Keeps original CLI of pmc_crawler.py:
    --csv, --outdir, --delay, --timeout
- On first crawl, also extracts <figure> images/captions and writes into info.csv:
    figure_image_urls | figure_captions | figures_json
- Also writes papers/{PMCID}.md including a "Figures" section.

Output (info.csv) columns:
[PMCID, title, date, lead_author, pdf_href, abstract,
 figure_image_urls, figure_captions, figures_json]
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
import json

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36"
    )
}

# ---------------------- helpers ---------------------- #


def extract_year_month(ref_str: str) -> Dict[str, Optional[int]]:
    """Extract {year, month} from a citation-like string, e.g., '2021 Jan'."""
    try:
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
        next(r, None)  # header
        for row in r:
            if not row or not any((cell or "").strip() for cell in row):
                continue
            total += 1
    return total


def _clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def _sanitize_for_csv_soft(s: Optional[str]) -> str:
    """Soft sanitize (used for captions/URLs). Keep commas as CSV handles quoting."""
    if s is None:
        return ""
    # remove nulls only
    return s.replace("\u0000", "").strip()


def _sanitize_for_csv_title_abs(s: str) -> str:
    """Historical sanitize from original script for title/abstract (kept for backward compatibility)."""
    return _clean_text((s or "").replace(",", " "))


def _safe_select_text(soup: BeautifulSoup, selector: str) -> str:
    el = soup.select_one(selector)
    return _clean_text(el.get_text()) if el else ""

# ---------------------- figure parsing (merged from new script) ---------------------- #


def _pick_img_src(img) -> str:
    for key in ("src", "data-src"):
        val = (img.get(key) or "").strip()
        if val:
            return val
    srcset = (img.get("srcset") or "").strip()
    if srcset:
        candidates = [p.strip().split(" ")[0]
                      for p in srcset.split(",") if p.strip()]
        if candidates:
            return candidates[-1]
    return ""


def _find_article_container(soup: BeautifulSoup):
    candidates = [
        {"id": "article-container"},
        {"id": "maincontent"},
        {"id": "article"},
        {"class_": "article-container"},
        {"id": "content"},
        {"role": "main"},
    ]
    for kw in candidates:
        found = soup.find(True, **kw)
        if found:
            return found
    return soup


def _extract_figures(container, base_url: str) -> List[Dict[str, object]]:
    figures: List[Dict[str, object]] = []
    blocks = list(container.find_all("figure"))
    blocks.extend(container.find_all(
        "div", class_=lambda c: c and "fig" in c.split()))

    for fig in blocks:
        label = ""
        candidates = [
            fig.find("figcaption"),
            fig.find(["h3", "h4", "h5", "header"]),
            fig.find("span", class_=lambda c: c and "fig-label" in c.split()),
        ]
        for node in candidates:
            if node:
                label = _clean_text(node.get_text(" "))
                if label:
                    break

        img_urls: List[str] = []
        for img in fig.find_all("img"):
            src = _pick_img_src(img)
            if src:
                img_urls.append(urljoin(base_url, src))
        if not img_urls:
            for a in fig.find_all("a", href=True):
                href = a["href"].strip()
                if not href:
                    continue
                low = href.lower()
                if any(low.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp")) or "tileshop_pmc" in low:
                    img_urls.append(urljoin(base_url, href))

        caption = ""
        fc = fig.find("figcaption")
        if fc:
            caption = _clean_text(fc.get_text(" "))
        else:
            alt = fig.find("div", class_=lambda c: c and "caption" in c.split()) or \
                fig.find("p", class_=lambda c: c and "caption" in c.split())
            if alt:
                caption = _clean_text(alt.get_text(" "))
            else:
                texts = []
                for p in fig.find_all("p"):
                    if p.find("img"):
                        continue
                    t = _clean_text(p.get_text(" "))
                    if t:
                        texts.append(t)
                if texts:
                    caption = " ".join(texts)

        dedup_urls = list(dict.fromkeys(img_urls))
        figures.append(
            {"label": label, "images": dedup_urls, "caption": caption})

    return figures


def parse_figures_from_soup(soup: BeautifulSoup, base_url: str) -> Tuple[str, str, str, List[Dict[str, object]]]:
    container = _find_article_container(soup)
    figures = _extract_figures(container, base_url)

    all_img_urls: List[str] = []
    all_captions: List[str] = []
    for f in figures:
        all_img_urls.extend(f.get("images", []))
        cap = (f.get("caption") or "").strip()
        if cap:
            all_captions.append(cap)

    figure_image_urls_str = " | ".join(list(dict.fromkeys(all_img_urls)))
    figure_captions_str = " || ".join(all_captions)
    figures_json_str = json.dumps(figures, ensure_ascii=False)
    return figure_image_urls_str, figure_captions_str, figures_json_str, figures

# ---------------------- parsing main fields ---------------------- #


def parse_article_html(html: str, base_url: str) -> Dict[str, object]:
    """Parse a PMC article HTML and extract metadata + markdown + figures."""
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one("#article-container") or soup

    # Title
    title_el = container.select_one("h1")
    title = _clean_text(title_el.get_text()) if title_el else _clean_text(
        (soup.find("meta", {"name": "citation_title"}) or {}).get("content")
    )

    # First author
    author_el = container.select_one(
        ".front-matter .name.western") or container.select_one(".name.western")
    first_author = _clean_text(author_el.get_text()) if author_el else ""

    # PMCID
    text_all = container.get_text(" ", strip=True)
    m = re.search(r"PMCID:\s*(PMC\d+)", text_all)
    pmcid = m.group(1) if m else None
    if not pmcid:
        m2 = re.search(r"/articles/(PMC\d+)/", base_url)
        pmcid = m2.group(1) if m2 else ""

    # Date (YYYY.MM)
    citation_text = _safe_select_text(soup, ".pmc-layout__citation")
    date_info = extract_year_month(citation_text) if citation_text else {}
    date_str = f"{date_info.get('year', '')}.{date_info.get('month', '')}" if date_info else ""

    # Abstract
    abstract = ""
    abs_sec = container.select_one(
        "section.abstract") or container.select_one("#abstract1")
    if abs_sec:
        ps = abs_sec.find_all(["p", "div"])
        abstract = _clean_text("  \n\n".join(
            _clean_text(p.get_text(" ")) for p in ps))

    # PDF href
    pdf_href = ""
    a = soup.select_one('a[data-ga-label="pdf_download_desktop"]') or \
        soup.select_one('a[data-ga-label*="pdf_download"]') or \
        soup.find("a", attrs={"title": "Download PDF"})
    if a and a.get("href"):
        pdf_href = urljoin(base_url, a["href"])
    else:
        meta_pdf = soup.find("meta", {"name": "citation_pdf_url"})
        if meta_pdf and meta_pdf.get("content"):
            pdf_href = meta_pdf.get("content")

    # Authors (all)
    author_all = [_clean_text(e.get_text())
                  for e in container.select(".front-matter .name.western")]
    if not author_all and first_author:
        author_all = [first_author]

    # -------- Figures --------
    fig_urls_str, fig_caps_str, figs_json_str, figs = parse_figures_from_soup(
        soup, base_url)

    # Markdown body
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

    # Append Figures section to markdown
    # if figs:
    #     md_lines.append("## Figures")
    #     for f in figs:
    #         label = f.get("label") or ""
    #         caption = f.get("caption") or ""
    #         images = f.get("images") or []
    #         if label:
    #             md_lines.append(f"**{label}**")
    #         for img in images:
    #             md_lines.append(f"![{_clean_text(label or 'figure')}]({img})")
    #         if caption:
    #             md_lines.append(caption)

    md_content = "\n\n".join(md_lines).strip()

    return {
        "title": title or "",
        "date": date_str,
        "first_author": first_author or "",
        "pmcid": pmcid or "",
        "abstract": abstract or "",
        "pdf_href": pdf_href or "",
        "markdown": md_content,
        # figures
        "figure_image_urls": fig_urls_str,
        "figure_captions": fig_caps_str,
        "figures_json": figs_json_str,  # keep commas (do NOT sanitize/alter)
    }

# ---------------------- CSV header detection ---------------------- #


def detect_columns_from_csv(csv_path: str) -> Tuple[str, str]:
    """Detect (title_col, link_col) from CSV header, mapping to 'title' internally."""
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, [])
        headers_lc = [h.lower() for h in headers]

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
    rows_info: List[Dict[str, object]] = []

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
                parsed = parse_article_html(resp.text, url)
            except Exception as e:
                # on failure, keep columns consistent
                pmcid_guess = ""
                m = re.search(r"/articles/(PMC\d+)/", url)
                if m:
                    pmcid_guess = m.group(1)
                rows_info.append({
                    "PMCID": pmcid_guess,
                    "title": _sanitize_for_csv_title_abs(title_hint),
                    "date": "",
                    "lead_author": "",
                    "pdf_href": "",
                    "abstract": f"Fetch error: {e}",
                    "figure_image_urls": "",
                    "figure_captions": "",
                    "figures_json": "",
                })
                time.sleep(delay)
                continue

            # Write markdown
            pmcid = parsed.get("pmcid", "")
            md_filename = f"{pmcid}.md" if pmcid else f"NO_PMCID_{idx}.md"
            md_path = os.path.join(papers_dir, md_filename)
            os.makedirs(papers_dir, exist_ok=True)
            with open(md_path, "w", encoding="utf-8") as mf:
                mf.write(parsed.get("markdown", ""))

            # Accumulate info for info.csv
            output_title = parsed.get("title") or title_hint
            rows_info.append({
                "PMCID": pmcid,
                "title": _sanitize_for_csv_title_abs(output_title).replace("/", " "),
                "date": parsed.get("date", ""),
                "lead_author": _sanitize_for_csv_title_abs(parsed.get("first_author", "")),
                "pdf_href": (parsed.get("pdf_href", "") or "").replace(",", ""),
                "abstract": _sanitize_for_csv_title_abs(parsed.get("abstract", "")).replace("/", " "),
                # figures (soft sanitize; keep commas)
                "figure_image_urls": _sanitize_for_csv_soft(parsed.get("figure_image_urls", "")),
                "figure_captions": _sanitize_for_csv_soft(parsed.get("figure_captions", "")),
                # DO NOT sanitize/alter
                "figures_json": parsed.get("figures_json", ""),
            })

            time.sleep(delay)

    # Save info.csv (ordered columns)
    info_path = os.path.join(outdir, "info.csv")
    fieldnames = [
        "PMCID", "title", "date", "lead_author", "pdf_href", "abstract",
        "figure_image_urls", "figure_captions", "figures_json"
    ]
    with open(info_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for r in rows_info:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    # Post-process info.csv and prune orphaned markdown files
    df = pd.read_csv(info_path, dtype=str, keep_default_na=False)
    df = df.drop_duplicates().dropna(
        subset=["PMCID", "title"]).reset_index(drop=True)
    df["title"] = df["title"].astype(str).str.replace("/", " ", regex=False)
    df.to_csv(info_path, index=False, encoding="utf-8-sig")

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
        description="Crawl PMC article pages and extract metadata + markdown + figure info."
    )
    parser.add_argument("--csv", default="SB_publication_PMC.csv",
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
