import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text: str) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _top_terms_from_scores(scores, feature_names, top_n: int = 8) -> List[str]:
    scores = np.asarray(scores).ravel()
    if scores.size == 0:
        return []
    idx = scores.argsort()[::-1][:top_n]
    return [feature_names[i] for i in idx if scores[i] > 0]


def _novelty_label(score: float) -> str:
    if score >= 0.70:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"


def _gap_note(keywords: List[str], avg_cluster_similarity: float, cluster_size: int) -> str:
    if cluster_size <= 1:
        return "Only one paper in this cluster. Add more papers to compare themes."
    if avg_cluster_similarity >= 0.78:
        return "Theme is crowded. Narrow by dataset, method, population, or time period."
    if len(keywords) >= 2:
        return f"Possible gap around '{keywords[0]}' and '{keywords[1]}'."
    if len(keywords) == 1:
        return f"Possible gap around '{keywords[0]}'."
    return "Theme is broad. Use more specific keywords or a smaller dataset."


@dataclass
class AnalysisResult:
    dataframe: pd.DataFrame
    cluster_summaries: Dict[int, Dict[str, object]]
    global_keywords: List[str]
    silhouette: Optional[float]
    duplicate_pairs: List[Tuple[int, int, float]]


def _choose_cluster_count(X, n: int, max_clusters: int) -> Tuple[int, Optional[float]]:
    if n <= 2:
        return 1, None

    upper = min(max_clusters, n - 1)
    best_k = 1
    best_score = None

    for k in range(2, upper + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(X, labels)
        except Exception:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_k = k

    if best_k == 1:
        return 1, None
    return best_k, best_score


def analyze_abstracts(
    texts: List[str],
    titles: Optional[List[str]] = None,
    max_clusters: int = 5,
    duplicate_threshold: float = 0.92,
) -> AnalysisResult:
    records = []

    for i, text in enumerate(texts):
        cleaned = clean_text(text)
        if not cleaned:
            continue

        title = f"Paper {len(records) + 1}"
        if titles and i < len(titles):
            safe_title = str(titles[i]).strip()
            if safe_title:
                title = safe_title

        records.append(
            {
                "paper_id": len(records) + 1,
                "title": title,
                "text": cleaned,
            }
        )

    if not records:
        empty = pd.DataFrame()
        return AnalysisResult(empty, {}, [], None, [])

    df_in = pd.DataFrame(records)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        min_df=1,
    )
    X = vectorizer.fit_transform(df_in["text"].tolist())
    feature_names = vectorizer.get_feature_names_out()

    global_scores = np.asarray(X.mean(axis=0)).ravel()
    global_keywords = _top_terms_from_scores(global_scores, feature_names, top_n=12)

    n = len(df_in)

    if n == 1:
        labels = np.array([0])
        silhouette = None
    else:
        best_k, silhouette = _choose_cluster_count(X, n=n, max_clusters=max_clusters)
        if best_k == 1:
            labels = np.zeros(n, dtype=int)
        else:
            labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X)

    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0.0)

    novelty_scores = 1 - sim.max(axis=1) if n > 1 else np.array([1.0])
    duplicate_pairs: List[Tuple[int, int, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= duplicate_threshold:
                duplicate_pairs.append(
                    (
                        int(df_in.iloc[i]["paper_id"]),
                        int(df_in.iloc[j]["paper_id"]),
                        float(sim[i, j]),
                    )
                )

    rows = []
    cluster_summaries: Dict[int, Dict[str, object]] = {}

    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        cluster_df = df_in[mask]
        cluster_matrix = X[mask]

        cluster_scores = np.asarray(cluster_matrix.mean(axis=0)).ravel()
        cluster_keywords = _top_terms_from_scores(cluster_scores, feature_names, top_n=8)
        theme = ", ".join(cluster_keywords[:3]) if cluster_keywords else f"Cluster {cluster_id}"

        sub_sim = sim[np.ix_(mask, mask)]
        avg_cluster_similarity = float(sub_sim[sub_sim > 0].mean()) if np.any(sub_sim > 0) else 0.0

        if len(cluster_df) > 1:
            internal_scores = np.asarray(cluster_matrix @ cluster_matrix.T).ravel()
            rep_idx_local = int(np.argmax(sub_sim.mean(axis=1)))
            representative_row = cluster_df.iloc[rep_idx_local]
        else:
            representative_row = cluster_df.iloc[0]

        cluster_summaries[int(cluster_id)] = {
            "theme": theme,
            "keywords": cluster_keywords,
            "size": int(len(cluster_df)),
            "avg_similarity": round(avg_cluster_similarity, 3),
            "gap": _gap_note(cluster_keywords, avg_cluster_similarity, len(cluster_df)),
            "representative_title": representative_row["title"],
        }

    for idx in range(n):
        cluster_id = int(labels[idx])
        cluster_info = cluster_summaries[cluster_id]

        best_match_idx = int(sim[idx].argmax()) if n > 1 else idx
        best_match_score = float(sim[idx].max()) if n > 1 else 0.0
        best_match_title = df_in.iloc[best_match_idx]["title"] if n > 1 and best_match_score > 0 else "—"

        novelty = float(novelty_scores[idx])
        duplicate_flag = bool(best_match_score >= duplicate_threshold and best_match_idx != idx)

        rows.append(
            {
                "paper_id": int(df_in.iloc[idx]["paper_id"]),
                "title": df_in.iloc[idx]["title"],
                "cluster": cluster_id,
                "theme": cluster_info["theme"],
                "novelty_score": round(novelty, 3),
                "novelty_level": _novelty_label(novelty),
                "duplicate_flag": duplicate_flag,
                "best_match_title": best_match_title,
                "best_match_score": round(best_match_score, 3),
                "gap_note": cluster_info["gap"],
                "abstract": df_in.iloc[idx]["text"][:300] + ("..." if len(df_in.iloc[idx]["text"]) > 300 else ""),
            }
        )

    out_df = pd.DataFrame(rows)

    return AnalysisResult(
        dataframe=out_df,
        cluster_summaries=cluster_summaries,
        global_keywords=global_keywords,
        silhouette=round(float(silhouette), 3) if silhouette is not None else None,
        duplicate_pairs=duplicate_pairs,
    )


def build_markdown_report(result: AnalysisResult) -> str:
    df = result.dataframe
    lines = []
    lines.append("# Research Gap Radar Report")
    lines.append("")
    lines.append(f"- Papers analysed: **{len(df)}**")
    lines.append(f"- Themes found: **{df['cluster'].nunique() if not df.empty else 0}**")
    lines.append(
        f"- Silhouette score: **{result.silhouette}**"
        if result.silhouette is not None
        else "- Silhouette score: **N/A**"
    )
    lines.append(f"- Global keywords: {', '.join(result.global_keywords[:12]) if result.global_keywords else 'None'}")
    lines.append("")

    lines.append("## Cluster summaries")
    for cid, info in result.cluster_summaries.items():
        lines.append(f"### Cluster {cid}")
        lines.append(f"- Theme: {info['theme']}")
        lines.append(f"- Keywords: {', '.join(info['keywords']) if info['keywords'] else 'None'}")
        lines.append(f"- Representative paper: {info['representative_title']}")
        lines.append(f"- Gap note: {info['gap']}")
        lines.append("")

    lines.append("## Paper-level results")
    if not df.empty:
        for _, row in df.iterrows():
            lines.append(
                f"- Paper {row['paper_id']} ({row['title']}): "
                f"cluster={row['cluster']}, novelty={row['novelty_score']}, "
                f"level={row['novelty_level']}, duplicate={row['duplicate_flag']}"
            )

    return "\n".join(lines)