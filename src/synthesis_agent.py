"""
Agent 4: Gemini Literature Synthesizer Agent.
Uses Google Gemini (Gemini 1.5 Flash) to synthesize research clusters,
novelty gaps, and paper abstracts into an academic literature review report.
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load .env file if available
load_dotenv()


def generate_gemini_synthesis(
    topic: str,
    cluster_summaries: Dict[int, Dict[str, object]],
    top_novel_papers: List[Dict[str, object]],
    global_keywords: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    model_name: str = "gemini-1.5-flash",
) -> str:
    """
    Agent 4: Synthesizes research clusters and novelty gaps using Google Gemini.

    Args:
        topic: The user's research topic or query phrase.
        cluster_summaries: Dictionary of cluster ID -> theme, keywords, size, avg_similarity, gap, representative_title.
        top_novel_papers: List of top outlier/novel paper records.
        global_keywords: List of leading domain terms across the entire corpus.
        api_key: Google Gemini API key (defaults to GEMINI_API_KEY env variable).
        model_name: Gemini model to use (default: gemini-1.5-flash).

    Returns:
        Structured Markdown literature synthesis report.
    """
    active_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()

    # Build structured analytical context from Agents 2 & 3
    context_blocks = []
    context_blocks.append(f"### Research Domain: {topic}")
    
    if global_keywords:
        context_blocks.append(f"Global Domain Keywords: {', '.join(global_keywords[:10])}")
        
    context_blocks.append("\n### Identified Thematic Clusters:")
    for cid, info in cluster_summaries.items():
        kw = ", ".join(info.get("keywords", [])[:5]) if info.get("keywords") else "N/A"
        context_blocks.append(
            f"- **Cluster {cid} — {info.get('theme')}**\n"
            f"  * Size: {info.get('size')} papers | Internal Similarity: {info.get('avg_similarity', 0.0):.3f}\n"
            f"  * Key Terminology: {kw}\n"
            f"  * Representative Anchor: \"{info.get('representative_title')}\"\n"
            f"  * Algorithmic Gap Detection: {info.get('gap')}"
        )

    context_blocks.append("\n### High-Novelty Outlier Papers:")
    for p in top_novel_papers[:6]:
        title = p.get("title", "Untitled")
        novelty = p.get("novelty_score", 0.0)
        theme = p.get("theme", "N/A")
        context_blocks.append(f"- **{title}** (Novelty: {novelty:.3f}, Cluster: {theme})")

    structured_context = "\n".join(context_blocks)

    # If no API key is provided, return rich heuristic report
    if not active_key:
        return _build_heuristic_fallback(topic, structured_context)

    # Call Google Gemini API
    try:
        import google.generativeai as genai

        genai.configure(api_key=active_key)

        system_prompt = (
            "You are a Senior Principal AI Academic Researcher and Literature Synthesis Specialist. "
            "You are given structured algorithmic findings from a research paper corpus: thematic clusters, "
            "cluster density, novelty scores, and identified gap notes.\n\n"
            "Your task is to write a rigorous, publication-grade Literature Review Synthesis Report.\n"
            "Organize your report with clear markdown headers:\n"
            "# Literature Review Synthesis & Gap Analysis\n"
            "## 1. Executive Landscape & State of the Art\n"
            "## 2. Core Thematic Pillars (Synthesize the identified clusters and their interplay)\n"
            "## 3. High-Novelty Frontiers & Outlier Contributions\n"
            "## 4. Critical Literature Gaps & Research White Spaces (Where the field is missing research)\n"
            "## 5. Strategic Agenda for Future Work (Actionable research questions to pursue)\n\n"
            "Tone: Rigorous, insightful, authoritative, and academic. Do not use generic filler."
        )

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
        )

        user_prompt = (
            f"Please synthesize the following research landscape for '{topic}':\n\n"
            f"{structured_context}"
        )

        response = model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=3000,
            ),
        )

        if response and response.text:
            return response.text
        return _build_heuristic_fallback(topic, structured_context, error="Empty response from Gemini API.")

    except Exception as e:
        return _build_heuristic_fallback(topic, structured_context, error=str(e))


def _build_heuristic_fallback(topic: str, context: str, error: Optional[str] = None) -> str:
    """Fallback generator when no API key is provided or if network fails."""
    lines = []
    lines.append(f"# 📄 Algorithmic Literature Synthesis: {topic}")
    lines.append("")
    if error:
        lines.append(f"> ⚠️ **Gemini Notice**: Falling back to heuristic synthesis ({error}).")
    else:
        lines.append("> 💡 **Tip**: Enter your **Google Gemini API Key** in the sidebar to generate deep neural synthesis.")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append(f"This literature corpus for **{topic}** was partitioned through unsupervised semantic clustering and pairwise novelty estimation.")
    lines.append("")
    lines.append("## 2. Thematic Decomposition & Landscape")
    lines.append(context)
    lines.append("")
    lines.append("## 3. Identified Gaps & Strategic Outlook")
    lines.append("- Dense clusters indicate saturated methodological approaches where incremental papers compete for novelty.")
    lines.append("- High-novelty outliers represent under-explored frontier combinations that provide strong candidate topics for future investigation.")
    lines.append("")
    return "\n".join(lines)
