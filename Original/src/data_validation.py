from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


COLUMN_ALIASES = {
    "kab_kota_pt": "kabupaten_pt",
    "kabupaten_pt": "kabupaten_pt",
    "unnamed_7": "jumlah_lulusan",
}


def normalize_column_name(name: object) -> str:
    text = unicodedata.normalize("NFKC", str(name))
    text = text.strip().lower()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return COLUMN_ALIASES.get(text, text)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_column_name(col) for col in out.columns]
    return out


def normalize_text_key(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", " ", text).strip().casefold()
    text = re.sub(r"^provinsi\s+", "prov. ", text)
    text = re.sub(r"^kabupaten\s+", "kab. ", text)
    return text or pd.NA


def normalize_category(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", " ", text).strip().upper()
    return text or pd.NA


def require_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise AssertionError(f"{label}: kolom wajib tidak ditemukan: {missing}")


def add_keys(
    df: pd.DataFrame,
    *,
    include_gender: bool = False,
    include_level: bool = False,
    include_stem: bool = False,
) -> pd.DataFrame:
    require_columns(df, ["provinsi_pt", "kabupaten_pt"], "add_keys")
    out = df.copy()
    out["_prov_key"] = out["provinsi_pt"].map(normalize_text_key)
    out["_kab_key"] = out["kabupaten_pt"].map(normalize_text_key)

    if include_gender:
        require_columns(out, ["jk"], "add_keys(gender)")
        out["_jk_key"] = out["jk"].map(normalize_category)

    if include_level:
        require_columns(out, ["nm_jenj_didik"], "add_keys(level)")
        out["_jenjang_key"] = out["nm_jenj_didik"].map(normalize_category)

    if include_stem:
        require_columns(out, ["a_stem"], "add_keys(stem)")
        stem = out["a_stem"].map(normalize_category)
        stem = stem.replace({"NON-STEM": "NON STEM", "NONSTEM": "NON STEM"})
        out["_stem_key"] = stem

    return out


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num.div(den.where(den.ne(0)))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_excel_flexible(
    path: Path,
    required_columns: Sequence[str],
    *,
    max_header_row: int = 6,
) -> pd.DataFrame:
    engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
    errors: list[str] = []

    for header_row in range(max_header_row + 1):
        try:
            frame = pd.read_excel(path, header=header_row, engine=engine)
            frame = standardize_columns(frame)
            frame = frame.dropna(axis=1, how="all")
            if set(required_columns).issubset(frame.columns):
                return frame
        except ImportError as exc:
            if path.suffix.lower() == ".xls":
                raise ImportError(
                    "File .xls memerlukan xlrd>=2.0.1. "
                    "Jalankan: pip install xlrd>=2.0.1"
                ) from exc
            raise
        except Exception as exc:
            errors.append(f"header={header_row}: {type(exc).__name__}: {exc}")

    raise ValueError(
        f"Gagal menemukan header untuk {path.name}. "
        f"Kolom wajib: {list(required_columns)}. "
        f"Percobaan: {' | '.join(errors[-3:])}"
    )


def assert_unique_keys(df: pd.DataFrame, keys: Sequence[str], label: str) -> pd.DataFrame:
    duplicated = df.loc[df.duplicated(list(keys), keep=False)].sort_values(list(keys))
    if not duplicated.empty:
        raise AssertionError(
            f"{label}: ditemukan {len(duplicated):,} baris pada kunci duplikat {list(keys)}."
        )
    return duplicated


def collapse_repeated_metrics(
    df: pd.DataFrame,
    keys: Sequence[str],
    metrics: Sequence[str],
    label: str,
) -> pd.DataFrame:
    require_columns(df, list(keys) + list(metrics), label)
    check = df.groupby(list(keys), dropna=False)[list(metrics)].nunique(dropna=False)
    inconsistent = check.gt(1).any(axis=1)
    if inconsistent.any():
        bad = check.loc[inconsistent]
        raise AssertionError(
            f"{label}: indikator tingkat wilayah tidak konsisten pada "
            f"{int(inconsistent.sum())} kunci. Contoh:\n{bad.head()}"
        )
    return df[list(keys) + list(metrics)].drop_duplicates(list(keys))


def reconcile_metrics(
    source: pd.DataFrame,
    merged: pd.DataFrame,
    *,
    keys: Sequence[str],
    metric_pairs: Sequence[tuple[str, str]],
    source_label: str,
    atol: float = 1e-9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert_unique_keys(source, keys, f"{source_label}—source")
    assert_unique_keys(merged, keys, f"{source_label}—merged")

    joined = source.merge(
        merged,
        how="outer",
        on=list(keys),
        indicator=True,
        suffixes=("_source", "_merged"),
    )

    reports = []
    discrepancies = []

    for source_metric, merged_metric in metric_pairs:
        s_col = (
            source_metric
            if source_metric not in merged.columns
            else f"{source_metric}_source"
        )
        m_col = (
            merged_metric
            if merged_metric not in source.columns
            else f"{merged_metric}_merged"
        )

        both = joined["_merge"].eq("both")
        s = pd.to_numeric(joined.loc[both, s_col], errors="coerce")
        m = pd.to_numeric(joined.loc[both, m_col], errors="coerce")
        diff = s - m
        mismatch_mask = ~(np.isclose(s, m, atol=atol, rtol=0, equal_nan=True))

        source_only = joined["_merge"].eq("left_only")
        merged_only = joined["_merge"].eq("right_only")

        reports.append(
            {
                "source": source_label,
                "source_metric": source_metric,
                "merged_metric": merged_metric,
                "source_keys": len(source),
                "merged_keys": len(merged),
                "compared_keys": int(both.sum()),
                "source_only_keys": int(source_only.sum()),
                "merged_only_keys": int(merged_only.sum()),
                "mismatch_keys": int(mismatch_mask.sum()),
                "max_abs_difference": (
                    float(diff.abs().max()) if len(diff) and diff.notna().any() else 0.0
                ),
                "source_total_all_keys": float(
                    pd.to_numeric(source[source_metric], errors="coerce").sum()
                ),
                "source_total_compared_keys": float(s.sum()),
                "merged_total_compared_keys": float(m.sum()),
                "delta_compared": float(diff.sum()),
                "status": "PASS" if not mismatch_mask.any() else "FAIL",
            }
        )

        if mismatch_mask.any():
            bad = joined.loc[both].loc[mismatch_mask].copy()
            bad["reconciliation_source"] = source_label
            bad["reconciliation_metric"] = f"{source_metric} -> {merged_metric}"
            bad["difference"] = diff.loc[mismatch_mask].to_numpy()
            discrepancies.append(bad)

    report_df = pd.DataFrame(reports)
    discrepancy_df = (
        pd.concat(discrepancies, ignore_index=True)
        if discrepancies
        else pd.DataFrame()
    )
    return report_df, discrepancy_df
