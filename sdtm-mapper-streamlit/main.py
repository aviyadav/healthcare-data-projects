#!/usr/bin/env python3
"""
Streamlit UI for SDTM Mapper
---------------------------
Upload your SDTM metadata CSV and your non-standard variables CSV, tweak parameters,
run the mapping, and download results.
"""

import io
from pathlib import Path
import pandas as pd
import streamlit as st


# Reuse core functions from the user's script by copying minimal logic here
# to avoid import path issues with Downloads folder copies.

import math
import re
from collections import Counter
from functools import lru_cache


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in {"dataset name","dataset","domain","table","table name"}:
            mapping[c] = "Dataset name"
        elif lc in {"variable label","label"}:
            mapping[c] = "Variable Label"
        elif lc in {"variable name","name","varname","var name"}:
            mapping[c] = "Variable Name"
        elif lc in {"role","roles"}:
            mapping[c] = "Role"
        elif lc in {"cdisc notes","notes","description","cdisc note"}:
            mapping[c] = "CDISC Notes"
        else:
            mapping[c] = c
    return df.rename(columns=mapping)


def normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common input headers to expected names for mapping."""
    mapping = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in {"dataset name","dataset","domain","table","table name"}:
            mapping[c] = "Dataset name"
        elif lc in {"variable label (non-standard)","variable label","label"}:
            mapping[c] = "Variable Label (non-standard)"
        elif lc in {"variable name (non-standard)","variable name","variable","name","varname","var name"}:
            mapping[c] = "Variable Name (non-standard)"
        elif lc in {"description","notes","note","details","cdisc notes"}:
            mapping[c] = "Description"
        else:
            mapping[c] = c
    return df.rename(columns=mapping)


def concat_search_text(row: pd.Series) -> str:
    return ("{} {} {} {} {}".format(
        row.get("Variable Name",""),
        row.get("Variable Label",""),
        row.get("CDISC Notes",""),
        row.get("Role",""),
        row.get("Dataset name","")
    )).lower()


def build_tfidf_corpus(texts, ngram_range=(1,2), min_df=1):
    tokenized = []
    for t in texts:
        ss = str(t).lower()
        tokens = re.findall(r"[a-z0-9]+", ss)
        grams = []
        for n in range(ngram_range[0], ngram_range[1]+1):
            grams += ["_".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        tokenized.append(grams)
    dfreq = Counter()
    for grams in tokenized:
        dfreq.update(set(grams))
    N = len(tokenized)
    vocab = {g:i for i,g in enumerate(sorted(dfreq.keys())) if dfreq[g] >= min_df}
    idf = {g: math.log((N+1)/(dfreq[g]+1)) + 1.0 for g in vocab}
    rows = []
    norms = []
    for grams in tokenized:
        tf = Counter([g for g in grams if g in vocab])
        row = {g: tf[g]*idf[g] for g in tf}
        norm = (sum(v*v for v in row.values()))**0.5 or 1.0
        rows.append(row)
        norms.append(norm)
    return vocab, idf, rows, norms


def vectorize(text, vocab, idf, ngram_range=(1,2)):
    s = str(text).lower()
    tokens = re.findall(r"[a-z0-9]+", s)
    grams = []
    for n in range(ngram_range[0], ngram_range[1]+1):
        grams += ["_".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    tf = Counter([g for g in grams if g in vocab])
    row = {g: tf[g]*idf[g] for g in tf}
    norm = (sum(v*v for v in row.values()))**0.5 or 1.0
    return row, norm


def cosine_sim_row_to_corpus(query_row, query_norm, corpus_rows, corpus_norms):
    sims = []
    qkeys = set(query_row.keys())
    for i, row in enumerate(corpus_rows):
        keys = qkeys & row.keys()
        dot = sum(query_row[k]*row[k] for k in keys)
        sims.append(dot / (query_norm * corpus_norms[i]))
    return sims


def map_variables(input_df: pd.DataFrame, meta_df: pd.DataFrame, vocab, idf, meta_rows, meta_norms, top_k=3, domain_strict=False):
    results = []
    for idx, row in input_df.fillna("").iterrows():
        ds = str(row.get("Dataset name","")).strip()
        # Build richer query text using variable label/name + description + dataset
        qtext = "{} {} {} {}".format(
            row.get("Variable Label (non-standard)",""),
            row.get("Variable Name (non-standard)",""),
            row.get("Description",""),
            ds,
        )

        pool = meta_df
        if ds and "Dataset name" in meta_df.columns:
            mask = meta_df["Dataset name"].astype(str).str.upper().eq(ds.upper())
            if mask.any():
                pool = meta_df[mask]
            elif domain_strict:
                continue

        qrow, qnorm = vectorize(qtext, vocab, idf, ngram_range=(1,2))

        pool_idx = pool.index.tolist()
        rows = [meta_rows[i] for i in pool_idx]
        norms = [meta_norms[i] for i in pool_idx]
        sims = cosine_sim_row_to_corpus(qrow, qnorm, rows, norms)

        order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        for rank, oi in enumerate(order, start=1):
            mi = pool_idx[oi]
            m = meta_df.loc[mi]
            results.append({
                "input_row": idx,
                "input_dataset": ds,
                "input_variable_label_nonstandard": row.get("Variable Label (non-standard)",""),
                "input_variable_name_nonstandard": row.get("Variable Name (non-standard)",""),
                "input_description": row.get("Description",""),
                "match_rank": rank,
                "match_dataset": str(m.get("Dataset name","")),
                "match_variable_label": str(m.get("Variable Label","")),
                "match_variable_name": str(m.get("Variable Name","")),
                "match_role": str(m.get("Role","")),
                "match_notes": str(m.get("CDISC Notes","")),
                "similarity": round(sims[oi], 6)
            })
    return pd.DataFrame(results)


def best_match(matches_df: pd.DataFrame, threshold: float | None):
    if matches_df.empty:
        return matches_df
    top1 = matches_df.sort_values(["input_row","match_rank"]).groupby("input_row", as_index=False).first()
    if threshold is not None:
        top1 = top1[top1["similarity"] >= threshold].copy()
    return top1[[
        "input_row",
        "input_dataset",
        "input_variable_label_nonstandard",
        "input_variable_name_nonstandard",
        "input_description",
        "match_dataset",
        "match_variable_label",
        "match_variable_name",
        "match_role",
        "match_notes",
        "similarity",
    ]]


def merge_best_with_input(input_df: pd.DataFrame, best_df: pd.DataFrame) -> pd.DataFrame:
    if input_df.empty or best_df.empty:
        return input_df.copy()
    tmp = input_df.copy().reset_index().rename(columns={"index":"input_row"})
    merged = tmp.merge(best_df, on="input_row", how="left")
    cols_front = [
        "Dataset name",
        "Variable Label (non-standard)",
        "Variable Name (non-standard)",
        "Description",
        "match_variable_name",
        "match_variable_label",
        "match_role",
        "match_notes",
        "match_dataset",
        "similarity",
    ]
    cols_front = [c for c in cols_front if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in cols_front and c != "input_row"]
    return merged[cols_front + other_cols]


def apply_custom_theme(*args, **kwargs):
    """No-op to restore Streamlit's default theme."""
    return


# -------------------------------------------------------------
# Improved mapping utilities: deterministic lookup + TF-IDF fallback
# -------------------------------------------------------------

def normalize_key(text: str) -> str:
    """Normalize a text key for strict matching: lowercase, alnum-only."""
    s = re.sub(r"[^a-z0-9]+", "", str(text).lower())
    return s


# Minimal synonym map for common non-standard names → standard SDTM names
NONSTANDARD_TO_STANDARD = {
    "gender": "sex",
    "labtest": "lbtest",
}


def apply_synonym(name: str) -> str:
    key = normalize_key(name)
    return NONSTANDARD_TO_STANDARD.get(key, name)


def build_meta_indices(meta_df: pd.DataFrame):
    """Precompute fast lookup indices for name/label with and without domain scoping."""
    df = meta_df.copy()
    df["__name_key__"] = df.get("Variable Name", "").astype(str).map(normalize_key)
    df["__label_key__"] = df.get("Variable Label", "").astype(str).map(normalize_key)
    df["__domain_key__"] = df.get("Dataset name", "").astype(str).str.upper().str.strip()

    name_to_rows: dict[str, list[int]] = {}
    label_to_rows: dict[str, list[int]] = {}
    domain_name_to_rows: dict[str, list[int]] = {}
    domain_label_to_rows: dict[str, list[int]] = {}

    for i, r in df.iterrows():
        nk = r["__name_key__"]
        lk = r["__label_key__"]
        dk = r["__domain_key__"]
        if nk:
            name_to_rows.setdefault(nk, []).append(i)
            if dk:
                domain_name_to_rows.setdefault(f"{dk}::{nk}", []).append(i)
        if lk:
            label_to_rows.setdefault(lk, []).append(i)
            if dk:
                domain_label_to_rows.setdefault(f"{dk}::{lk}", []).append(i)

    return {
        "df": df,
        "name_to_rows": name_to_rows,
        "label_to_rows": label_to_rows,
        "domain_name_to_rows": domain_name_to_rows,
        "domain_label_to_rows": domain_label_to_rows,
    }


def resolve_by_indices(input_row: pd.Series, indices, meta_df: pd.DataFrame):
    if not indices:
        return None
    # If multiple, prefer the first for now. Could add smarter tie-breakers later.
    mi = indices[0]
    return meta_df.loc[mi]


def map_row_deterministic(input_row: pd.Series, meta_index: dict):
    """Try domain-scoped exact name/label matches first, with synonym expansion."""
    meta_df = meta_index["df"]
    domain = str(input_row.get("Dataset name", "")).upper().strip()
    name_raw = str(input_row.get("Variable Name (non-standard)", ""))
    label_raw = str(input_row.get("Variable Label (non-standard)", ""))

    # Candidate keys include raw and synonym-applied forms
    name_candidates = [name_raw, apply_synonym(name_raw)]
    label_candidates = [label_raw, apply_synonym(label_raw)]

    # Normalize to keys
    name_keys = [normalize_key(s) for s in name_candidates if s]
    label_keys = [normalize_key(s) for s in label_candidates if s]

    # Domain-scoped name match
    for nk in name_keys:
        if domain and nk:
            rows = meta_index["domain_name_to_rows"].get(f"{domain}::{nk}")
            resolved = resolve_by_indices(input_row, rows, meta_df)
            if resolved is not None:
                return resolved

    # Global name match
    for nk in name_keys:
        rows = meta_index["name_to_rows"].get(nk)
        resolved = resolve_by_indices(input_row, rows, meta_df)
        if resolved is not None:
            return resolved

    # Domain-scoped label match
    for lk in label_keys:
        if domain and lk:
            rows = meta_index["domain_label_to_rows"].get(f"{domain}::{lk}")
            resolved = resolve_by_indices(input_row, rows, meta_df)
            if resolved is not None:
                return resolved

    # Global label match
    for lk in label_keys:
        rows = meta_index["label_to_rows"].get(lk)
        resolved = resolve_by_indices(input_row, rows, meta_df)
        if resolved is not None:
            return resolved

    return None


def map_variables_v2(
    input_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    meta_index: dict,
    vocab,
    idf,
    meta_rows,
    meta_norms,
    desc_vocab,
    desc_idf,
    desc_rows,
    desc_norms,
    top_k: int = 3,
    domain_strict: bool = False,
    enforce_desc_overlap: bool = False,
    min_desc_overlap: int = 1,
    description_first: bool = True,
    description_min_similarity: float | None = None,
):
    """Deterministic matching first, then TF-IDF fallback for suggestions."""
    results = []
    for idx, row in input_df.fillna("").iterrows():
        ds = str(row.get("Dataset name", "")).strip()

        # 1) Description-first mapping against CDISC Notes
        desc = str(row.get("Description", "")).strip()
        if description_first and desc:
            # Optionally domain-scope the pool
            pool = meta_df
            if ds and "Dataset name" in meta_df.columns:
                mask = meta_df["Dataset name"].astype(str).str.upper().eq(ds.upper())
                if mask.any():
                    pool = meta_df[mask]
                elif domain_strict:
                    pool = meta_df.iloc[0:0]  # empty

            if len(pool) > 0:
                qrow_d, qnorm_d = vectorize(desc, desc_vocab, desc_idf, ngram_range=(1,2))
                pool_idx = pool.index.tolist()
                rows_d = [desc_rows[i] for i in pool_idx]
                norms_d = [desc_norms[i] for i in pool_idx]
                sims_d = cosine_sim_row_to_corpus(qrow_d, qnorm_d, rows_d, norms_d)
                if sims_d:
                    order_d = sorted(range(len(sims_d)), key=lambda i: sims_d[i], reverse=True)[:top_k]
                    best_sim = sims_d[order_d[0]] if order_d else 0.0
                    # If a threshold is set, enforce it
                    if description_min_similarity is None or best_sim >= float(description_min_similarity):
                        for rank, oi in enumerate(order_d, start=1):
                            mi = pool_idx[oi]
                            m = meta_df.loc[mi]
                            results.append({
                                "input_row": idx,
                                "input_dataset": ds,
                                "input_variable_label_nonstandard": row.get("Variable Label (non-standard)", ""),
                                "input_variable_name_nonstandard": row.get("Variable Name (non-standard)", ""),
                                "input_description": row.get("Description", ""),
                                "match_rank": rank,
                                "match_dataset": str(m.get("Dataset name", "")),
                                "match_variable_label": str(m.get("Variable Label", "")),
                                "match_variable_name": str(m.get("Variable Name", "")),
                                "match_role": str(m.get("Role", "")),
                                "match_notes": str(m.get("CDISC Notes", "")),
                                "similarity": round(sims_d[oi], 6),
                            })
                        # If description-first produced results, skip further steps
                        continue

        # 2) Deterministic resolution (name/label/synonyms)
        resolved = map_row_deterministic(row, meta_index)
        if resolved is not None:
            results.append({
                "input_row": idx,
                "input_dataset": ds,
                "input_variable_label_nonstandard": row.get("Variable Label (non-standard)", ""),
                "input_variable_name_nonstandard": row.get("Variable Name (non-standard)", ""),
                "input_description": row.get("Description", ""),
                "match_rank": 1,
                "match_dataset": str(resolved.get("Dataset name", "")),
                "match_variable_label": str(resolved.get("Variable Label", "")),
                "match_variable_name": str(resolved.get("Variable Name", "")),
                "match_role": str(resolved.get("Role", "")),
                "match_notes": str(resolved.get("CDISC Notes", "")),
                "similarity": 1.0,
            })
            continue

        # 3) TF-IDF fallback (existing behavior), optionally domain-scoped
        pool = meta_df
        if ds and "Dataset name" in meta_df.columns:
            mask = meta_df["Dataset name"].astype(str).str.upper().eq(ds.upper())
            if mask.any():
                pool = meta_df[mask]
            elif domain_strict:
                # No candidates in this domain and strict mode enabled
                continue

        qtext = "{} {} {} {}".format(
            row.get("Variable Label (non-standard)", ""),
            row.get("Variable Name (non-standard)", ""),
            row.get("Description", ""),
            ds,
        )

        qrow, qnorm = vectorize(qtext, vocab, idf, ngram_range=(1,2))

        pool_idx = pool.index.tolist()
        rows = [meta_rows[i] for i in pool_idx]
        norms = [meta_norms[i] for i in pool_idx]
        sims = cosine_sim_row_to_corpus(qrow, qnorm, rows, norms)

        order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        for rank, oi in enumerate(order, start=1):
            mi = pool_idx[oi]
            m = meta_df.loc[mi]
            # Optional description relationship check (token overlap)
            if enforce_desc_overlap:
                inp_desc = str(row.get("Description", ""))
                meta_desc = str(m.get("CDISC Notes", ""))
                inp_tokens = set(re.findall(r"[a-z0-9]+", inp_desc.lower()))
                meta_tokens = set(re.findall(r"[a-z0-9]+", meta_desc.lower()))
                if len(inp_tokens & meta_tokens) < int(min_desc_overlap):
                    continue
            results.append({
                "input_row": idx,
                "input_dataset": ds,
                "input_variable_label_nonstandard": row.get("Variable Label (non-standard)", ""),
                "input_variable_name_nonstandard": row.get("Variable Name (non-standard)", ""),
                "input_description": row.get("Description", ""),
                "match_rank": rank,
                "match_dataset": str(m.get("Dataset name", "")),
                "match_variable_label": str(m.get("Variable Label", "")),
                "match_variable_name": str(m.get("Variable Name", "")),
                "match_role": str(m.get("Role", "")),
                "match_notes": str(m.get("CDISC Notes", "")),
                "similarity": round(sims[oi], 6),
            })

    return pd.DataFrame(results)


st.set_page_config(page_title="SDTM Mapper", layout="wide")
apply_custom_theme()
st.title("SDTM Mapper - Streamlit UI")
st.write("Use the default project metadata or upload one-time metadata, then upload a non-standard file to map. Supports CSV, Excel, JSON, Parquet, TSV/TXT.")


DEFAULT_META_PATH = Path(__file__).parent / "metadata" / "default_metadata.csv"
DEFAULT_META_PATH.parent.mkdir(parents=True, exist_ok=True)


with st.sidebar:
    st.header("Inputs")
    use_default_meta = st.checkbox("Use project default metadata", value=DEFAULT_META_PATH.exists())
    meta_file = None
    if not use_default_meta:
        meta_file = st.file_uploader("Standard/Metadata file (CSV/Excel/JSON/Parquet/TSV/TXT)", type=["csv","tsv","txt","xlsx","xls","json","parquet"]) 
    input_file = st.file_uploader("Non-standard file (CSV/Excel/JSON/Parquet/TSV/TXT)", type=["csv","tsv","txt","xlsx","xls","json","parquet"]) 
    topk = st.number_input("Top-k candidates", value=3, min_value=1, max_value=20, step=1)
    threshold = st.number_input("Similarity threshold (optional)", value=0.0, min_value=0.0, max_value=1.0, step=0.01)
    use_threshold = st.checkbox("Filter best-match by threshold", value=False)
    domain_strict = st.checkbox("Strict domain filtering (use Dataset name)", value=False)
    enforce_desc_overlap = st.checkbox("Require description overlap", value=False, help="Ensure input Description shares words with CDISC Notes")
    min_desc_overlap = st.number_input("Minimum overlapping words", value=1, min_value=1, max_value=10, step=1)
    show_matrix = st.checkbox("Show similarity matrix (debug)", value=False)
    require_approval = st.checkbox("Require manual approval of mappings", value=True)
    # Show which metadata will be used
    if use_default_meta:
        if DEFAULT_META_PATH.exists():
            st.caption(f"Using default metadata: {DEFAULT_META_PATH}")
        else:
            st.error(f"Default metadata not found at {DEFAULT_META_PATH}. Upload a metadata file or place one there.")
    elif meta_file is not None:
        st.caption(f"Using uploaded metadata: {meta_file.name}")


run = st.button(
    "Run Mapping",
    type="primary",
    disabled=not (input_file and (use_default_meta or (meta_file is not None)))
)

def read_any_table(uploaded_file) -> pd.DataFrame:
    name = (uploaded_file.name or "").lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if name.endswith(".tsv") or name.endswith(".tab"):
            return pd.read_csv(uploaded_file, sep="\t")
        if name.endswith(".txt"):
            # Try to sniff delimiter
            return pd.read_csv(uploaded_file, sep=None, engine="python")
        if name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file, engine="openpyxl")
        if name.endswith(".xls"):
            return pd.read_excel(uploaded_file)  # requires xlrd for .xls
        if name.endswith(".json"):
            try:
                return pd.read_json(uploaded_file, lines=True)
            except Exception:
                uploaded_file.seek(0)
                return pd.read_json(uploaded_file)
        if name.endswith(".parquet"):
            return pd.read_parquet(uploaded_file)
    except Exception as e:
        raise ValueError(f"Failed to read '{uploaded_file.name}': {e}")
    raise ValueError("Unsupported file type. Please upload CSV, TSV, TXT, XLSX/XLS, JSON, or Parquet.")


def read_any_path(path: Path) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".tsv") or p.endswith(".tab"):
        return pd.read_csv(path, sep="\t")
    if p.endswith(".txt"):
        return pd.read_csv(path, sep=None, engine="python")
    if p.endswith(".xlsx"):
        return pd.read_excel(path, engine="openpyxl")
    if p.endswith(".xls"):
        return pd.read_excel(path)
    if p.endswith(".json"):
        try:
            return pd.read_json(path, lines=True)
        except Exception:
            return pd.read_json(path)
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError("Unsupported file extension for default metadata.")


@lru_cache(maxsize=1)
def get_cached_meta_index(meta_hash: str):
    return None


can_reuse_cached = (
    "meta_df" in st.session_state and
    "input_df" in st.session_state and
    isinstance(st.session_state.get("meta_df"), pd.DataFrame) and
    isinstance(st.session_state.get("input_df"), pd.DataFrame)
)

should_compute = (run and input_file and (use_default_meta or meta_file)) or (not run and can_reuse_cached)

if should_compute:
    try:
        if run:
            if use_default_meta:
                if not DEFAULT_META_PATH.exists():
                    st.error(f"Default metadata not found at {DEFAULT_META_PATH}. Upload a metadata file or place one there.")
                    st.stop()
                meta_df = read_any_path(DEFAULT_META_PATH).fillna("")
            else:
                meta_df = read_any_table(meta_file).fillna("")
            meta_df = normalize_columns(meta_df).fillna("")
            st.session_state["meta_df"] = meta_df
        else:
            meta_df = st.session_state["meta_df"].copy()

        meta_df["__search_text__"] = meta_df.apply(concat_search_text, axis=1)
        vocab, idf, meta_rows, meta_norms = build_tfidf_corpus(meta_df["__search_text__"].tolist(), ngram_range=(1,2), min_df=1)
        # Description-only corpus (CDISC Notes)
        desc_texts = meta_df.get("CDISC Notes", "").astype(str).tolist()
        desc_vocab, desc_idf, desc_rows, desc_norms = build_tfidf_corpus(desc_texts, ngram_range=(1,2), min_df=1)

        if run:
            input_df = read_any_table(input_file).fillna("")
            input_df = normalize_input_columns(input_df).fillna("")
            st.session_state["input_df"] = input_df
        else:
            input_df = st.session_state["input_df"].copy()

        # Build indices for deterministic mapping and run the new pipeline
        meta_index = build_meta_indices(meta_df)
        matches_df = map_variables_v2(
            input_df,
            meta_df,
            meta_index,
            vocab,
            idf,
            meta_rows,
            meta_norms,
            desc_vocab,
            desc_idf,
            desc_rows,
            desc_norms,
            top_k=int(topk),
            domain_strict=bool(domain_strict),
            enforce_desc_overlap=bool(enforce_desc_overlap),
            min_desc_overlap=int(min_desc_overlap),
            description_first=True,
            description_min_similarity=float(threshold) if use_threshold else None,
        )

        st.success(f"Computed {len(matches_df)} candidate rows.")
        if show_matrix:
            st.subheader("Similarity matrix (candidates)")
            st.dataframe(matches_df, use_container_width=True)

        if use_threshold:
            best_df = best_match(matches_df, float(threshold))
        else:
            best_df = best_match(matches_df, None)

        # Manual approval flow
        if require_approval and not best_df.empty:
            st.subheader("Proposed mappings - Review & Approve")

            # Initialize approval state with defaults (approved=True)
            if "approvals" not in st.session_state:
                st.session_state["approvals"] = {}
            approvals = st.session_state["approvals"]

            approve_cols = [
                "input_row",
                "input_dataset",
                "input_variable_label_nonstandard",
                "input_variable_name_nonstandard",
                "match_variable_name",
                "match_variable_label",
                "match_dataset",
                "similarity",
            ]
            approve_cols = [c for c in approve_cols if c in best_df.columns]

            # Controls: bulk actions
            c1, c2, c3 = st.columns([1,1,6])
            with c1:
                if st.button("Approve all"):
                    for r in best_df["input_row"].tolist():
                        approvals[r] = True
            with c2:
                if st.button("Reject all"):
                    for r in best_df["input_row"].tolist():
                        approvals[r] = False

            # Display rows with a Review popover per row
            for _, r in best_df[approve_cols].iterrows():
                rid = int(r["input_row"]) if "input_row" in r else _
                current = approvals.get(rid, True)
                st.markdown("---")
                row_cols = st.columns([5,5,4,4,2])
                with row_cols[0]:
                    st.write(f"Input: {r.get('input_variable_name_nonstandard','')} | {r.get('input_variable_label_nonstandard','')}")
                with row_cols[1]:
                    st.write(f"Match: {r.get('match_variable_name','')} | {r.get('match_variable_label','')}")
                with row_cols[2]:
                    st.write(f"Dataset: {r.get('input_dataset','')} → {r.get('match_dataset','')}")
                with row_cols[3]:
                    st.write(f"Similarity: {r.get('similarity',0)}")
                with row_cols[4]:
                    with st.popover("Review"):
                        st.write("Toggle to approve or reject")
                        decided = st.toggle(
                            f"Approve mapping for row {rid}",
                            value=bool(current),
                            key=f"toggle_{rid}",
                        )
                        approvals[rid] = decided
                        st.write("Status:", "Approved" if decided else "Rejected")

            # Persist approvals and build final
            st.session_state["approvals"] = approvals
            approved_rows = [rid for rid, ok in approvals.items() if ok]
            best_approved = best_df[best_df["input_row"].isin(approved_rows)].copy()
            final_df = merge_best_with_input(input_df, best_approved)
        else:
            # No approval required; use best matches directly
            final_df = merge_best_with_input(input_df, best_df)

        st.subheader("Final converted variables")
        st.dataframe(final_df, use_container_width=True)

        # Single download: final converted variables file
        with io.BytesIO() as buf:
            final_df.to_csv(buf, index=False)
            buf.seek(0)
            st.download_button(
                label="Download final.csv",
                data=buf.getvalue(),
                file_name="final.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Error: {e}")