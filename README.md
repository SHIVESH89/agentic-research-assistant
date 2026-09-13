# 🔬 AI Agentic Workflows for Research: Literature Radar & Synthesis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Google-Gemini](https://img.shields.io/badge/Gemini-1.5%20Flash-8E75B2.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary

Navigating vast scientific domains requires synthesizing hundreds of papers, identifying thematic consensus, and uncovering under-explored research white spaces. 

This project implements an autonomous **Multi-Agent Research Intelligence System** that:
1. **Agent 1 (Ingestion):** Ingests live scientific publications directly from the **arXiv API** (or curated offline domain datasets).
2. **Agent 2 (Clustering):** Performs **unsupervised semantic clustering** with dynamic Silhouette score optimization to automatically discover latent research themes.
3. **Agent 3 (Novelty Radar):** Quantifies **paper novelty scores** ($1 - \max(\text{similarity})$) and flags dense ("crowded") vs. sparse ("frontier") thematic areas.
4. **Visual Landscape:** Projects research papers onto an interactive **2D PCA Semantic Landscape** scaled by novelty.
5. **Agent 4 (Synthesis):** Employs **Google Gemini (Gemini 1.5 Flash)** to synthesize a publication-grade Literature Review report complete with identified white spaces and future research directions.

---

## 🏗️ Multi-Agent Architecture

```mermaid
flowchart TD
    A["User Query / Seed Topic"] --> B["Agent 1: arXiv Ingestion Agent"]
    B --> C["Agent 2: Semantic Clustering Agent"]
    C --> D["Agent 3: Novelty & Gap Radar Agent"]
    D --> E["Agent 4: Gemini Literature Synthesizer"]
    
    subgraph UI ["Streamlit Interactive Dashboard"]
        F["2D PCA Semantic Landscape"]
        G["Theme & Gap Breakdown"]
        H["Top Novelty Frontier Outliers"]
        I["Live Gemini Literature Review"]
    end
    
    C --> F
    D --> G
    D --> H
    E --> I
```

---

## 🤖 The Multi-Agent Pipeline

* **Agent 1: Ingestion Agent (`src/arxiv_agent.py`)**  
  Queries the Open Access arXiv API to retrieve real-world papers with titles, abstracts, authors, publication dates, and PDF links. Includes automatic fallback to curated datasets during network timeouts.
* **Agent 2: Semantic Clustering Agent (`research_gap.py`)**  
  Transforms abstracts into TF-IDF representations and evaluates candidate cluster numbers ($k \in [2, 10]$), selecting the optimal $k$ by maximizing the **Silhouette Score**.
* **Agent 3: Novelty & Gap Radar Agent (`research_gap.py`)**  
  Computes pairwise Cosine distances across the entire corpus. Calculates a normalized novelty score for each paper, flags redundant duplicates, and surfaces high-novelty outliers.
* **Agent 4: Literature Synthesis Agent (`src/synthesis_agent.py`)**  
  Feeds the algorithmic cluster breakdown, representative anchor papers, and detected white spaces into **Google Gemini (Gemini 1.5 Flash)** to draft a structured 5-section literature review survey.

---

## 📂 Project Structure

```text
├── src/
│   ├── __init__.py               # Package initializer
│   ├── arxiv_agent.py            # Agent 1: Real-world arXiv literature ingestion
│   └── synthesis_agent.py        # Agent 4: Google Gemini literature synthesis
├── dataset/
│   └── sample_gut_heart_papers.csv # 15 Curated real-world test papers
├── app.py                        # Interactive Streamlit dashboard & landscape visualizer
├── research_gap.py               # Algorithmic clustering, novelty & gap engine
├── requirements.txt              # Core dependencies
├── .gitignore                    # Prevents .env and secrets from being committed
├── .env.example                  # Environment variable template
└── README.md                     # Comprehensive project documentation
```

---

## 🚀 Quick Start (Local)

### 1. Clone & Navigate
```bash
git clone https://github.com/SHIVESH89/ai-agentic-research-workflow.git
cd ai-agentic-research-workflow
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. *(Runs immediately out-of-the-box with built-in analytical engines).*

---

## 📊 Sample Research Queries
- `Gut microbiome cardiovascular disease`
- `Antimicrobial resistance livestock metagenomics`
- `Agentic AI workflow orchestration`
- `RNA therapeutics delivery mechanisms`
