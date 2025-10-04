# uv add transformers adapter-transformers torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from adapters import AutoAdapterModel
import torch
import numpy as np
import pandas as pd
import os
import json


def l2_normalize(a, axis=1, eps=1e-12):
    norm = np.linalg.norm(a, axis=axis, keepdims=True)
    return a / (norm + eps)


class PaperRetriever:
    """
    - 최초 실행 시 info.csv를 임베딩해 인덱스를 저장하고,
      이후에는 저장된 인덱스를 불러와 빠르게 검색함.
    - 검색은 Numpy 기반 코사인 유사도(FAISS 미사용)로 수행하여 libomp 충돌을 회피함.
      - 논문의 개수는 529개로 FAISS를 사용하는 것이 오버해드가 더 크기에 미사용.
      - (옵션) 초기 벡터 검색 상위 N(기본 50)개를 Cross-Encoder로 재랭킹.

    Args:
        csv_path (str): 논문 메타데이터 CSV 경로. 
        store_dir (str): 인덱스(임베딩)를 저장할 경로. 기본 ".rag_store"
        papers_dir (str): 전체 논문 텍스트(.md)가 위치한 디렉터리. 기본 "papers"
        cross_encoder_model (str): HF 허브 Cross-Encoder 모델 이름/경로. 기본 "cross-encoder/ms-marco-MiniLM-L-6-v2"
    """

    def __init__(
        self,
        csv_path: str,
        store_dir: str = ".rag_store",
        papers_dir: str = "papers",
        device: str = "cpu",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.csv_path = csv_path
        self.store_dir = store_dir
        csv_dir = os.path.dirname(os.path.abspath(self.csv_path)) or "."
        if os.path.isabs(papers_dir):
            self.papers_dir = papers_dir
        else:
            self.papers_dir = os.path.join(csv_dir, papers_dir)
        self.device = device
        self.cross_encoder_model = cross_encoder_model
        self.cross_encoder = None  # Lazy-load for reranking
        self.ce_tok = None
        self.ce_model = None

        os.makedirs(self.store_dir, exist_ok=True)

        # 모델/토크나이저 로드 (Hugging Face Hub)
        # SPECTER2 토크나이저
        self.tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
        # 어댑터 지원 베이스 모델
        self.model = AutoAdapterModel.from_pretrained(
            "allenai/specter_plus_plus")
        # 문서 임베딩용(Proximity)과 질의 임베딩용(Query) 어댑터를 각각 로드
        self.model.load_adapter("allenai/specter2", load_as="proximity")
        self.model.load_adapter(
            "allenai/specter2_adhoc_query", load_as="query")
        self.model.eval()

        # CSV 로드
        self.papers_df = pd.read_csv(self.csv_path)
        self.papers = self.papers_df.to_dict("records")
        self._paper_text_cache = {}

        # 인덱스 로드 또는 생성
        self.index = None  # L2-normalized doc embeddings
        self._build_or_load_index()

    # ---------- 내부 유틸 ----------
    def _isEnglish(self, s: str) -> bool:
        try:
            s.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        return True

    def _translate2EN(self, query: str) -> str:
        return query  # 영어가 아니라면 영어로 변역해서 입력하는 것이 적합.

    def _cfg_path(self) -> str:
        return os.path.join(self.store_dir, "cfg.json")

    def _index_path(self) -> str:
        return os.path.join(self.store_dir, "doc_index.npy")

    def _paper_file_path(self, pmcid):
        if not pmcid:
            return None
        pmcid_str = str(pmcid).strip()
        if not pmcid_str:
            return None
        return os.path.join(self.papers_dir, f"{pmcid_str}.md")

    def _get_paper_text(self, pmcid):
        if not pmcid:
            return ""
        pmcid_str = str(pmcid).strip()
        if not pmcid_str:
            return ""
        if pmcid_str in self._paper_text_cache:
            return self._paper_text_cache[pmcid_str]
        path = self._paper_file_path(pmcid_str)
        if not path or not os.path.exists(path):
            self._paper_text_cache[pmcid_str] = ""
            return ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except OSError:
            text = ""
        self._paper_text_cache[pmcid_str] = text
        return text

    def _papers_signature(self):
        if not os.path.isdir(self.papers_dir):
            return None
        count = 0
        total_size = 0
        latest_mtime = 0
        try:
            for entry in os.scandir(self.papers_dir):
                if not entry.is_file():
                    continue
                if not entry.name.lower().endswith(".md"):
                    continue
                stat = entry.stat()
                count += 1
                total_size += int(stat.st_size)
                latest_mtime = max(latest_mtime, int(stat.st_mtime))
        except FileNotFoundError:
            return None
        return {
            "path": os.path.abspath(self.papers_dir),
            "count": count,
            "size": total_size,
            "mtime": latest_mtime,
        }

    def _paper_to_text(self, p: dict) -> str:
        """Join title and full body (fallback: abstract) for Cross-Encoder input."""
        title = (p.get("title") or "").strip()
        body = self._get_paper_text(p.get("PMCID"))
        if not body:
            body = (p.get("abstract") or "").strip()
        if not title and not body:
            return ""
        if title and body:
            return f"{title} {self.tok.sep_token} {body}"
        return title or body

    def _csv_signature(self):
        """CSV 변경 여부 확인용 간단한 시그니처."""
        try:
            st = os.stat(self.csv_path)
            return {
                "path": os.path.abspath(self.csv_path),
                "size": int(st.st_size),
                "mtime": float(st.st_mtime),
                "rows": int(len(self.papers)),
            }
        except FileNotFoundError:
            return None

    def _build_or_load_index(self):
        cfg_path = self._cfg_path()
        idx_path = self._index_path()
        sig = self._csv_signature()
        paper_sig = self._papers_signature()

        # 저장된 인덱스 로드 시도
        if os.path.exists(cfg_path) and os.path.exists(idx_path):
            try:
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
                idx = np.load(idx_path).astype("float32")

                expected = {k: sig[k]
                            for k in ("path", "size", "rows")} if sig else None
                papers_expected = paper_sig
                cfg_papers_sig = cfg.get("papers_signature")
                if (
                    expected is not None
                    and cfg.get("csv_signature", {}) == expected
                    and cfg_papers_sig == papers_expected
                    and idx.shape[0] == len(self.papers)
                ):
                    self.index = idx
                    print("[RAG] Loaded existing index from store.")
                    return
            except Exception:
                pass  # 손상되었거나 호환 불가인 경우 재생성

        # 인덱스 재생성
        print("[RAG] Building index from CSV...")
        doc_emb = self.encode_papers(self.papers)
        self.index = l2_normalize(doc_emb, axis=1)

        # 저장
        np.save(idx_path, self.index)
        if sig is None:
            raise FileNotFoundError(
                f"CSV metadata not found at {self.csv_path}")
        cfg = {
            "csv_signature": {k: sig[k] for k in ("path", "size", "rows")},
            "papers_signature": paper_sig,
            "dim": int(self.index.shape[1]),
        }
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
        print("[RAG] Index built and saved.")

    # ---------- 임베딩 ----------
    @torch.no_grad()
    def encode_papers(self, papers, batch_size: int = 32, max_length: int = 512):
        self.model.set_active_adapters("proximity")
        self.model.to(self.device)
        out = []
        for i in range(0, len(papers), batch_size):
            batch = papers[i: i + batch_size]
            texts = []
            for p in batch:
                title = (p.get("title") or "").strip()
                body = self._get_paper_text(p.get("PMCID"))
                if not body:
                    body = (p.get("abstract") or "").strip()
                if title and body:
                    texts.append(f"{title}{self.tok.sep_token}{body}")
                else:
                    texts.append(title or body)
            enc = self.tok(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_token_type_ids=False,
            ).to(self.device)
            last_hidden = self.model(**enc).last_hidden_state  # [B, L, H]
            cls = last_hidden[:, 0, :]  # [B, H] -> 768-d
            out.append(cls.cpu().numpy().astype("float32"))
        return np.vstack(out) if out else np.empty((0, 768), dtype="float32")

    @torch.no_grad()
    def encode_queries(self, queries, max_length: int = 64):
        self.model.set_active_adapters("query")
        self.model.to(self.device)
        enc = self.tok(
            queries,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.device)
        q = self.model(
            **enc).last_hidden_state[:, 0, :].cpu().numpy().astype("float32")
        return l2_normalize(q, axis=1)

    def _load_cross_encoder(self):
        if self.ce_model is None:
            self.ce_tok = AutoTokenizer.from_pretrained(
                self.cross_encoder_model)
            self.ce_model = AutoModelForSequenceClassification.from_pretrained(
                self.cross_encoder_model)
            self.ce_model.to(self.device)
            self.ce_model.eval()

    @torch.no_grad()
    def _ce_predict(self, pairs, batch_size: int = 32, max_length: int = 512):
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            texts = [p[0] for p in batch]
            docs = [p[1] for p in batch]
            enc = self.ce_tok(
                texts,
                text_pair=docs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            out = self.ce_model(**enc)
            logits = out.logits  # [B, 1] or [B, 2]
            if logits.shape[-1] == 1:
                s = logits.squeeze(-1).detach().cpu().numpy()
            else:
                s = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            scores.extend(s.tolist())
        return np.array(scores, dtype="float32")

    # ---------- 검색 ----------
    def search(self, query: str, k: int = 5, ce_rerank_k: int = 50):
        if self.index is None:
            self._build_or_load_index()

        if not self._isEnglish(query):
            query = self._translate2EN(query)

        # 1) Dense retrieval with SPECTER2 query encoder
        q = self.encode_queries([query])  # [1, 768]
        scores = np.dot(self.index, q[0])  # cosine similarity (정규화된 내적)

        # 가져올 후보 개수 (재랭킹 후보는 기본 50, 최소 k)
        n_candidates = max(
            k, ce_rerank_k if ce_rerank_k and ce_rerank_k > 0 else k)
        n_candidates = min(n_candidates, scores.shape[0])

        # Top-n 후보 고르기 (partial sort)
        idx_part = np.argpartition(scores, -n_candidates)[-n_candidates:]
        idx_sorted_dense = idx_part[np.argsort(scores[idx_part])[::-1]]

        # 2) (옵션) Cross-Encoder 재랭킹
        if ce_rerank_k and ce_rerank_k > 0:
            self._load_cross_encoder()
            pairs = [(query, self._paper_to_text(self.papers[int(j)]))
                     for j in idx_sorted_dense]
            ce_scores = self._ce_predict(pairs, batch_size=32)

            order = np.argsort(ce_scores)[::-1]
            final_idx = [int(idx_sorted_dense[o]) for o in order[:k]]
            results = []
            for rank, j in enumerate(final_idx, start=1):
                rec = dict(self.papers[int(j)])
                rec["score"] = float(scores[int(j)])      # dense score
                # cross-encoder score
                rec["ce_score"] = float(ce_scores[order[rank - 1]])
                rec["rank"] = rank
                results.append(rec)
            return results

        # 3) 재랭킹 비활성화 시: dense 결과 상위 k개 반환
        final_idx = idx_sorted_dense[:k]
        results = []
        for rank, j in enumerate(final_idx, start=1):
            rec = dict(self.papers[int(j)])
            rec["score"] = float(scores[int(j)])
            rec["rank"] = rank
            results.append(rec)
        return results


if __name__ == "__main__":
    import time

    # 기본 경로들(모두 변경 가능)
    rag = PaperRetriever(
        csv_path="info.csv",
        store_dir=".rag_store",
        papers_dir="papers",
    )

    # 예시 질의
    q = "The impact of solar wind on humans in space"
    hits = rag.search(q, k=5, ce_rerank_k=50)
    for r in hits:
        if "ce_score" in r:
            print(
                f"[dense={r.get('score'):.4f} | ce={r.get('ce_score'):.4f}] {r.get('PMCID')} {r.get('title')}")
        else:
            print(
                f"[dense={r.get('score'):.4f}] {r.get('PMCID')} {r.get('title')}")

    # 이미 로드된 상태에서는 얼마나 빠른가보자.
    time.sleep(1)
    print("\nQuestion: Microgravity will dilate human blood vessels and promote growth.\n\n")

    q = "Microgravity will dilate human blood vessels and promote growth."
    hits = rag.search(q, k=5, ce_rerank_k=50)
    print([r.get('PMCID') for r in hits])
    for r in hits:
        if "ce_score" in r:
            print(
                f"[dense={r.get('score'):.4f} | ce={r.get('ce_score'):.4f}] {r.get('PMCID')} {r.get('title')}")
        else:
            print(
                f"[dense={r.get('score'):.4f}] {r.get('PMCID')} {r.get('title')}")

"""
[dense=0.7909 | ce=1.7362] PMC11988870 Microgravity and Cellular Biology: Insights into Cellular Responses and Implications for Human Health
[dense=0.7515 | ce=0.4570] PMC7787258 Prolonged Exposure to Microgravity Reduces Cardiac Contractility and Initiates Remodeling in Drosophila
[dense=0.7518 | ce=-0.3368] PMC4110898 Fifteen Days Microgravity Causes Growth in Calvaria of Mice
[dense=0.7859 | ce=-2.1094] PMC7339929 Simulated Microgravity Induces Regionally Distinct Neurovascular and Structural Remodeling of Skeletal Muscle and Cutaneous Arteries in the Rat
[dense=0.7582 | ce=-2.3219] PMC6275019 Synergistic Effects of Weightlessness  Isoproterenol  and Radiation on DNA Damage Response and Cytokine Production in Immune Cells
"""

"""
[dense=0.7984 | ce=0.6348] PMC11988870 Microgravity and Cellular Biology: Insights into Cellular Responses and Implications for Human Health
[dense=0.7716 | ce=-0.8358] PMC4110898 Fifteen Days Microgravity Causes Growth in Calvaria of Mice
[dense=0.7627 | ce=-0.8487] PMC5515531 Exposure of Mycobacterium marinum to low-shear modeled microgravity: effect on growth  the transcriptome and survival under stress
[dense=0.7663 | ce=-0.8712] PMC7787258 Prolonged Exposure to Microgravity Reduces Cardiac Contractility and Initiates Remodeling in Drosophila
[dense=0.7824 | ce=-1.1669] PMC5460135 Investigation of simulated microgravity effects on Streptococcus mutans physiology and global gene expression
"""
