import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# Load local .env if present
load_dotenv()

# Import agents
from research_gap import analyze_abstracts, build_markdown_report
from src.arxiv_agent import fetch_arxiv_papers
from src.synthesis_agent import generate_gemini_synthesis

st.set_page_config(
    page_title="AI Agentic Workflow for Research",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AI Agentic Workflows for Research")
st.markdown(
    "**Multi-Agent Research Assistant**: Autonomous paper ingestion, unsupervised semantic clustering, "
    "novelty & literature gap radar, and publication-ready **Gemini AI literature review synthesis**."
)
st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("🔑 Google Gemini API")
    api_key_input = (os.getenv("GEMINI_API_KEY") or "").strip()
    st.success("api key : active")

    st.markdown("---")
    st.subheader("📥 Literature Ingestion")
    input_mode = st.radio(
        "Select source",
        ["Live arXiv Search", "Upload dataset", "Paste abstracts"],
        index=0
    )

    st.markdown("---")
    st.subheader("🔬 Clustering & Gap Parameters")
    max_clusters = st.slider("Max clusters (k)", 2, 10, 5)
    duplicate_threshold = st.slider("Duplicate threshold", 0.70, 0.99, 0.92, 0.01)

# --- INPUT AREA ---
texts = []
titles = []
authors_list = []
links_list = []
current_topic = "Literature Exploration"

if input_mode == "Live arXiv Search":
    c1, c2 = st.columns([3, 1])
    with c1:
        query = st.text_input(
            "Research Query / Domain Keyword",
            value="Gut microbiome cardiovascular disease",
            placeholder="e.g., Antimicrobial resistance livestock, Agentic AI, RNA therapeutics..."
        )
    with c2:
        paper_limit = st.number_input("Max Papers to Fetch", min_value=5, max_value=60, value=25, step=5)

    if st.button("🚀 Fetch & Analyze with Agents", type="primary"):
        current_topic = query
        with st.spinner(f"Agent 1 is querying arXiv API for '{query}'..."):
            try:
                arxiv_papers = fetch_arxiv_papers(query=query, max_results=int(paper_limit))
                if not arxiv_papers:
                    st.warning("No papers returned from arXiv for this query. Try different terms.")
                else:
                    st.session_state["papers_data"] = arxiv_papers
                    st.session_state["topic"] = query
                    st.rerun()
            except Exception as e:
                st.error(f"Error fetching from arXiv: {e}")

    if "papers_data" in st.session_state and st.session_state["papers_data"]:
        for p in st.session_state["papers_data"]:
            texts.append(p["abstract"])
            titles.append(p["title"])
            authors_list.append(p.get("authors", ""))
            links_list.append(p.get("link", ""))
        current_topic = st.session_state.get("topic", current_topic)

elif input_mode == "Upload dataset":
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                uploaded_df = pd.read_csv(uploaded_file)
            else:
                uploaded_df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            uploaded_df = None

        if uploaded_df is not None and not uploaded_df.empty:
            cols = list(uploaded_df.columns)
            c1, c2 = st.columns(2)
            with c1:
                title_col = st.selectbox("Title column (optional)", ["(none)"] + cols)
            with c2:
                text_cols = st.multiselect("Text/Abstract columns to analyze", cols, default=[cols[0]])

            if text_cols:
                for _, row in uploaded_df.iterrows():
                    combined = " ".join(str(row[c]) for c in text_cols if pd.notna(row[c]))
                    if combined.strip():
                        texts.append(combined)
                        if title_col != "(none)":
                            val = row[title_col]
                            titles.append(str(val) if pd.notna(val) else f"Paper {len(titles)+1}")
                        else:
                            titles.append(f"Paper {len(titles)+1}")
                        authors_list.append("")
                        links_list.append("")

    if st.button("Run Analysis on Uploaded Data", type="primary"):
        current_topic = uploaded_file.name if uploaded_file else "Uploaded Dataset"

else:
    pasted = st.text_area(
        "Enter abstracts (one per line)",
        height=180,
        placeholder="Abstract 1...\nAbstract 2...\nAbstract 3..."
    )
    if st.button("Run Analysis on Pasted Abstracts", type="primary"):
        if pasted.strip():
            lines = [line.strip() for line in pasted.split("\n") if line.strip()]
            texts = lines
            titles = [f"Paper {i+1}" for i in range(len(lines))]
            authors_list = ["" for _ in lines]
            links_list = ["" for _ in lines]
            current_topic = "Manual Input Collection"

# --- EXECUTION PIPELINE ---
if texts:
    with st.spinner("Agent 2 (Clustering) & Agent 3 (Novelty Radar) are analyzing the corpus..."):
        result = analyze_abstracts(
            texts=texts,
            titles=titles if titles else None,
            max_clusters=max_clusters,
            duplicate_threshold=duplicate_threshold,
        )

    df = result.dataframe.copy()

    if df.empty:
        st.warning("No valid text extracted.")
        st.stop()

    # Attach authors and links if available
    if len(authors_list) == len(df):
        df["authors"] = authors_list
        df["link"] = links_list

    # --- RESULTS DASHBOARD ---
    st.markdown("---")
    st.subheader(f"📊 Landscape Analysis: {current_topic}")

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ingested Papers", len(result.dataframe))
    m2.metric("Optimal Clusters", result.dataframe["cluster"].nunique())
    m3.metric("Duplicate / Redundant", int(result.dataframe["duplicate_flag"].sum()))
    m4.metric("High Novelty Outliers", int((result.dataframe["novelty_level"] == "High").sum()))

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tabs = st.tabs([
        "🌐 Overview & Insights",
        "🗺️ 2D Research Landscape",
        "📂 Thematic Clusters",
        "📑 Paper Inventory",
        "🤖 Gemini Literature Synthesis",
        "📥 Export Artifacts"
    ])

    # Tab 0: Overview
    with tabs[0]:
        st.write("**Top Global Domain Keywords:**")
        st.write(", ".join(result.global_keywords) if result.global_keywords else "None")
        st.markdown("<br>", unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            st.write("**Paper Distribution by Thematic Cluster**")
            cluster_counts = result.dataframe["cluster"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(cluster_counts.index.astype(str), cluster_counts.values, color="#1f77b4")
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.tick_params(left=False)
            ax.set_xlabel("Cluster ID")
            ax.set_ylabel("Paper Count")
            st.pyplot(fig, clear_figure=True)

        with right:
            st.write("**Novelty Score Distribution**")
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            ax2.hist(result.dataframe["novelty_score"], bins=10, color="#2ca02c", edgecolor="white")
            ax2.spines[['top', 'right', 'left']].set_visible(False)
            ax2.tick_params(left=False)
            ax2.set_xlabel("Novelty Score (0 = Redundant, 1 = Outlier)")
            st.pyplot(fig2, clear_figure=True)

        st.write("**Top Novel Frontier Papers:**")
        top_novel = result.dataframe.sort_values("novelty_score", ascending=False).head(8)
        cols_to_show = ["paper_id", "title", "theme", "novelty_score", "gap_note"]
        if "link" in top_novel.columns:
            cols_to_show.append("link")
        st.dataframe(top_novel[cols_to_show], use_container_width=True, hide_index=True)

    # Tab 1: 2D Research Landscape (PCA)
    with tabs[1]:
        st.write("### 2D Projection of Research Corpus")
        st.markdown("Papers are projected into 2D semantic space via PCA on TF-IDF embeddings.")
        if len(texts) >= 3:
            vec = TfidfVectorizer(stop_words="english", max_features=500)
            X_mat = vec.fit_transform(texts)
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_mat.toarray())
            
            pca_df = pd.DataFrame({
                "PCA1": coords[:, 0],
                "PCA2": coords[:, 1],
                "Cluster": result.dataframe["cluster"].astype(str),
                "Title": result.dataframe["title"],
                "Novelty": result.dataframe["novelty_score"]
            })

            fig_pca, ax_pca = plt.subplots(figsize=(8, 5))
            scatter = ax_pca.scatter(
                pca_df["PCA1"],
                pca_df["PCA2"],
                c=result.dataframe["cluster"],
                cmap="tab10",
                s=pca_df["Novelty"] * 120 + 40,
                alpha=0.8,
                edgecolors="black"
            )
            ax_pca.set_title(f"Semantic Map: {current_topic} (Dot size indicates novelty)")
            ax_pca.set_xlabel("Principal Component 1")
            ax_pca.set_ylabel("Principal Component 2")
            ax_pca.spines[['top', 'right']].set_visible(False)
            st.pyplot(fig_pca, clear_figure=True)
        else:
            st.info("Add 3 or more papers to generate the 2D Semantic Landscape.")

    # Tab 2: Clusters
    with tabs[2]:
        for cid, info in result.cluster_summaries.items():
            st.markdown(f"### Cluster {cid}: {info['theme']}")
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Keywords:** {', '.join(info['keywords']) if info['keywords'] else 'None'}")
                st.write(f"**Cluster Size:** {info['size']} papers")
                st.write(f"**Internal Density:** {info['avg_similarity']}")
            with c2:
                st.write(f"**Anchor Paper:** {info['representative_title']}")
                st.write(f"**Detected Gap Note:** {info['gap']}")

            cluster_rows = result.dataframe[result.dataframe["cluster"] == cid]
            st.dataframe(
                cluster_rows[["paper_id", "title", "novelty_score", "novelty_level"]],
                use_container_width=True,
                hide_index=True
            )
            st.markdown("---")

    # Tab 3: Detailed Data
    with tabs[3]:
        search_kw = st.text_input("Filter papers by keyword", "")
        display_df = df.copy()
        if search_kw:
            display_df = display_df[
                display_df["title"].str.contains(search_kw, case=False, na=False)
                | display_df["abstract"].str.contains(search_kw, case=False, na=False)
            ]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Tab 4: Gemini Literature Synthesis
    with tabs[4]:
        st.write("### 🤖 Agent 4: Google Gemini Literature Synthesis")
        st.markdown(
            "Agent 4 inspects the mathematical clusters, novelty scores, and identified white spaces, "
            "then synthesizes a structured academic literature review."
        )

        if st.button("✨ Synthesize Literature Review with Gemini", type="primary"):
            top_novel_list = result.dataframe.sort_values("novelty_score", ascending=False).to_dict("records")
            with st.spinner("Agent 4 (Gemini 1.5 Flash) is synthesizing the literature landscape..."):
                synthesis_md = generate_gemini_synthesis(
                    topic=current_topic,
                    cluster_summaries=result.cluster_summaries,
                    top_novel_papers=top_novel_list,
                    global_keywords=result.global_keywords,
                    api_key=api_key_input
                )
                st.session_state["synthesis_report"] = synthesis_md

        if "synthesis_report" in st.session_state:
            st.markdown(st.session_state["synthesis_report"])
            st.download_button(
                "📥 Download Synthesis Report (.md)",
                data=st.session_state["synthesis_report"].encode("utf-8"),
                file_name=f"literature_synthesis_{current_topic.replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True
            )

    # Tab 5: Export
    with tabs[5]:
        c1, c2 = st.columns(2)
        with c1:
            csv_bytes = result.dataframe.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Structured CSV Dataset",
                data=csv_bytes,
                file_name=f"research_landscape_{current_topic.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with c2:
            base_md = build_markdown_report(result).encode("utf-8")
            st.download_button(
                "📥 Download Algorithmic Summary (.md)",
                data=base_md,
                file_name="cluster_gap_summary.md",
                mime="text/markdown",
                use_container_width=True
            )