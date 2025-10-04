import os
from pathlib import Path
import shutil
import pandas as pd

INFO_PARQUET = Path("info.parquet")
JSON_DIR = Path("summary_results_json")
MD_DIR = Path("papers")
PMCID_COL = "PMCID"


def load_pmcids_from_df(path: Path, col: str) -> set[str]:
    df = pd.read_parquet(path)
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in {path}")
    # Normalize to trimmed strings
    return set(df[col].astype(str).str.strip())


def pmcids_from_dir(root: Path, target_suffix: str) -> set[str]:
    """Collect IDs from filenames (basename without extension), case-insensitive suffix match."""
    ids = set()
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() == target_suffix.lower():
            ids.add(p.stem.strip())
    return ids


def delete_files_by_ids(root: Path, ids_to_delete: set[str], suffix: str) -> list[Path]:
    deleted = []
    for _id in sorted(ids_to_delete):
        fp = root / f"{_id}{suffix}"
        try:
            if fp.exists():
                fp.unlink()
                deleted.append(fp)
        except Exception as e:
            print(f"[WARN] Failed to delete {fp}: {e}")
    return deleted


def main():
    # 1) Gather PMCID sets
    df_ids = load_pmcids_from_df(INFO_PARQUET, PMCID_COL)
    json_ids = pmcids_from_dir(JSON_DIR, ".json")
    md_ids = pmcids_from_dir(MD_DIR, ".md")

    # 2) Compute the intersection (the only IDs we keep everywhere)
    common = df_ids & json_ids & md_ids

    # 3) Determine removals
    drop_from_df = df_ids - common
    drop_json = json_ids - common
    drop_md = md_ids - common

    print("=== Counts BEFORE ===")
    print(f"DataFrame rows (unique PMCID): {len(df_ids)}")
    print(f"JSON files: {len(json_ids)}")
    print(f"MD files: {len(md_ids)}")
    print(f"Common PMCID across all 3: {len(common)}\n")

    # 4) Clean files first (JSON, MD)
    deleted_json = delete_files_by_ids(JSON_DIR, drop_json, ".json")
    deleted_md = delete_files_by_ids(MD_DIR, drop_md, ".md")

    # 5) Clean DataFrame (keep only common)
    df = pd.read_parquet(INFO_PARQUET)
    # Back up original parquet
    backup_path = INFO_PARQUET.with_suffix(INFO_PARQUET.suffix + ".bak")
    try:
        shutil.copy2(INFO_PARQUET, backup_path)
        print(f"[INFO] Backed up original parquet to: {backup_path}")
    except Exception as e:
        print(f"[WARN] Failed to back up parquet: {e}")

    df_clean = df[df[PMCID_COL].astype(str).str.strip().isin(common)].copy()
    df_clean.to_parquet(INFO_PARQUET, index=False)

    print("=== Removal Summary ===")
    print(f"Removed from JSON dir: {len(deleted_json)} files")
    if deleted_json:
        for p in deleted_json[:10]:
            print(f"  - {p}")
        if len(deleted_json) > 10:
            print(f"  ... (+{len(deleted_json)-10} more)")

    print(f"Removed from MD dir: {len(deleted_md)} files")
    if deleted_md:
        for p in deleted_md[:10]:
            print(f"  - {p}")
        if len(deleted_md) > 10:
            print(f"  ... (+{len(deleted_md)-10} more)")

    print(f"Removed from DataFrame: {len(drop_from_df)} PMCID values")

    # 6) Show AFTER counts
    df_ids_after = set(df_clean[PMCID_COL].astype(str).str.strip())
    json_ids_after = pmcids_from_dir(JSON_DIR, ".json")
    md_ids_after = pmcids_from_dir(MD_DIR, ".md")
    print("\n=== Counts AFTER ===")
    print(f"DataFrame rows (unique PMCID): {len(df_ids_after)}")
    print(f"JSON files: {len(json_ids_after)}")
    print(f"MD files: {len(md_ids_after)}")

    if not (df_ids_after == json_ids_after == md_ids_after == common):
        print("[WARN] Sets still differ after cleanup. Please inspect mismatches.")
    else:
        print("[OK] All three sources now have identical PMCID sets.")


if __name__ == "__main__":
    main()
