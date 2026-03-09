# Social Media Analysis

A data science project analysing Reddit discourse across three thematic areas: **AI-generated content**, **disinformation / fake news**, and **mental health & social media**. The project covers the full pipeline — from API data collection to semantic embeddings, LLM-based emotion analysis, and unsupervised topic modelling.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `task1.ipynb` | Data collection, visual exploration, semantic analysis & emotion detection |
| `task2.ipynb` | User participation analysis, sentiment impact & BERTopic topic modelling |

---

## Notebook 1 — Data Collection & Semantic Analysis (`task1.ipynb`)

### What I Did

1. **Reddit API data collection (PRAW)**  
   Discovered relevant subreddits by scanning keyword-tagged posts across three topics. Built a curated list of 24 subreddits and collected ~3,000 unique users and 476,842 activity records (posts + comments).

2. **Visual exploration**  
   - Interactive Plotly time-series chart of post volume per topic over time.  
   - User popularity dashboard combining total score, content count, and received comment count.

3. **Semantic UMAP map**  
   Encoded 1 500 Reddit messages with `all-mpnet-base-v2` (SentenceTransformers), reduced dimensionality with UMAP, and visualised clusters coloured by topic. *Finding: mental-health posts form the tightest semantic cluster; AI-content and disinformation clusters visibly overlap.*

4. **LLM emotion tagging**  
   Called the **CLARIN API** (Llama 3.3) to label each message with a fine-grained emotion (anger, joy, sadness, anxiety, …) and sentiment polarity.

5. **Emotion heatmap & spike detection**  
   Built a `subreddit × emotion` heatmap and implemented a statistical spike-detection algorithm. Automatically summarised what users discussed during each emotional spike using the LLM. *Finding: disinformation subreddits are dominated by anger; mental-health posts show the most diverse emotional profile.*

### What I Learned
- How to work responsibly with a rate-limited public API and design collection logic that stays within quota.
- How to represent text semantically and visualise high-dimensional embeddings in 2D.
- How to integrate a hosted LLM (CLARIN) into a data-science pipeline for structured annotation at scale.
- Negative/angry content tends to cluster together and stands out semantically, even before any labelling.

---

## Notebook 2 — User Participation & Topic Modelling (`task2.ipynb`)

### What I Did

1. **User activity analysis**  
   Measured the distribution of posts and comments per user per topic using box plots and summary statistics. Identified cross-topic users with a co-occurrence matrix. *Finding: activity follows a strong Pareto distribution — a small number of power users generate most content.*

2. **Sentiment impact on engagement**  
   Tested whether post sentiment (LLM labels + TextBlob + DistilBERT) predicts score and comment count using Kruskal-Wallis tests and violin plots. *Finding: negative posts receive up to 2× more upvotes than positive ones in disinformation and AI-content topics; mental-health is the exception where neutral content scores highest.*

3. **Sentiment model validation**  
   Compared three sentiment classifiers (CLARIN LLM, TextBlob, DistilBERT fine-tuned on SST-2) via confusion matrices to assess agreement and calibration.

4. **BERTopic topic modelling**  
   Trained a BERTopic model (`all-MiniLM-L6-v2` + UMAP + HDBSCAN + c-TF-IDF) on ~69 000 documents and discovered **29 micro-topics**. Visualised topic distributions globally and compared them between high- and low-popularity posts. Used Jensen-Shannon divergence to quantify topic-distribution differences.

5. **LLM-generated topic labels**  
   Automatically described the top 10 BERTopic topics by passing representative keywords and sample documents to the CLARIN LLM.

### What I Learned
- How to design a statistically rigorous sentiment-vs-engagement analysis, including choosing the right non-parametric tests for skewed data.
- How to build and tune a BERTopic pipeline end-to-end (embedding → UMAP → HDBSCAN → c-TF-IDF → LLM labelling).
- Cross-validating NLP models from different paradigms (lexicon-based, fine-tuned transformer, generative LLM) reveals both their individual biases and where they agree.
- Topic popularity is not random: certain micro-topics systematically attract higher engagement, providing actionable signals for content strategy or moderation.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Data collection | `praw`, Reddit API |
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn`, `plotly` |
| NLP / Embeddings | `sentence-transformers` (MPNet, MiniLM) |
| Dimensionality reduction | `umap-learn` |
| Clustering | `hdbscan` |
| Topic modelling | `bertopic` |
| Sentiment analysis | `textblob`, `transformers` (DistilBERT) |
| LLM integration | CLARIN API (Llama 3.3) |
| Statistics | `scipy` |

---

## 📊 Key Findings at a Glance

- **476,842** Reddit records collected across 3 topic areas from **~3,000 unique users**.
- **Negative content drives higher engagement** in AI and disinformation communities (up to 2× more upvotes).
- **Mental-health discourse is semantically cohesive** — the tightest cluster in UMAP space.
- **29 micro-topics** discovered automatically, with disinformation and AI topics showing stronger thematic diversity than mental-health content.
- Emotional spike analysis enables automatic event detection from social media without manual labelling.