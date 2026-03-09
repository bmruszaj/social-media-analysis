# Social Media Analysis

This project analyses Reddit discourse across three thematic areas — AI-generated content, disinformation and fake news, and mental health on social media. The work spans the full analytical pipeline, from API-based data collection through semantic text representation, LLM-assisted emotion annotation, and unsupervised topic modelling.

---

## Project Structure

| File | Description |
|------|-------------|
| `task1.ipynb` | Data collection, exploratory visualisation, semantic analysis, and emotion detection |
| `task2.ipynb` | User participation analysis, sentiment–engagement modelling, and BERTopic topic modelling |

---

## Notebook 1 — Data Collection and Semantic Analysis (`task1.ipynb`)

### What I Did

**Reddit data collection via PRAW.** I started by identifying relevant subreddits through keyword-based post scanning across the three topic areas. This produced a curated list of 24 subreddits, from which approximately 3,000 unique user accounts and 476,842 activity records (posts and comments) were collected.

**Exploratory visualisation.** I plotted post volume over time as an interactive Plotly time-series, broken down by topic. A separate user-level dashboard combined total score, content count, and received comment count into a composite popularity view.

**Semantic embedding and UMAP projection.** I encoded a sample of 1,500 messages with `all-mpnet-base-v2` (SentenceTransformers) and projected them into two dimensions using UMAP. The resulting scatter plot revealed that mental-health posts form a notably tight semantic cluster, while AI-content and disinformation posts overlap considerably in the embedding space.

**LLM-based emotion annotation.** I submitted each message to the CLARIN API (Llama 3.3) to obtain a fine-grained emotion label (anger, joy, sadness, anxiety, and others) as well as an overall sentiment polarity score.

**Emotion heatmap and temporal spike detection.** I built a subreddit-by-emotion frequency matrix and visualised it as a heatmap. A statistical spike-detection routine identified weeks with abnormal emotional intensity, and the LLM was used to summarise the content of those spikes automatically. Disinformation subreddits were consistently dominated by anger, whereas mental-health posts exhibited the most varied emotional profile across the full observation period.

### What I Learned

Working with the Reddit API at scale taught me how to design collection logic that respects rate limits while still gathering a statistically useful sample. Producing and interpreting UMAP projections gave me a clearer sense of what sentence-level embeddings actually capture about topical similarity. Integrating a generative LLM into the annotation pipeline — rather than relying solely on lexicon-based tools — showed how structured prompting can replace expensive manual labelling for large text corpora. One finding that stood out is that negatively toned content clusters distinctly in embedding space even before any sentiment label is applied, suggesting that tone is a primary axis of variation in these communities.

---

## Notebook 2 — User Participation and Topic Modelling (`task2.ipynb`)

### What I Did

**User activity analysis.** I examined the distribution of posts and comments per user per topic using box plots and descriptive statistics, and built a co-occurrence matrix to identify users active across multiple topic areas. Activity followed a strongly skewed distribution consistent with Pareto dynamics: a small proportion of users accounted for the majority of published content.

**Sentiment–engagement analysis.** I tested whether post sentiment predicts engagement (score and comment count) separately for each topic, using Kruskal-Wallis tests and violin plots. Negative posts attracted roughly twice as many upvotes as positive posts in the disinformation and AI-content communities. Mental health was the exception — neutral posts scored highest there, which aligns with the supportive rather than confrontational character of those communities.

**Sentiment model comparison.** I evaluated three sentiment classifiers side by side: labels from the CLARIN LLM, TextBlob polarity scores, and a DistilBERT model fine-tuned on SST-2. Confusion matrices were used to measure inter-method agreement and to identify systematic disagreements between the lexicon-based and transformer-based approaches.

**BERTopic topic modelling.** I assembled a BERTopic pipeline using `all-MiniLM-L6-v2` embeddings, UMAP for dimensionality reduction, HDBSCAN for clustering, and c-TF-IDF for topic representation. Applied to approximately 69,000 documents, the model produced 29 coherent micro-topics. I compared topic distributions between high-scoring and low-scoring posts using Jensen-Shannon divergence to identify topics associated with greater audience engagement.

**Automated topic labelling.** I described the top 10 BERTopic topics in plain language by submitting their representative keywords and sample documents to the CLARIN LLM, producing human-readable topic names without manual inspection.

### What I Learned

Designing the sentiment–engagement comparison required choosing the right statistical tests for highly skewed, non-normal data — Kruskal-Wallis rather than ANOVA — and interpreting effect sizes carefully rather than relying solely on p-values. Building the BERTopic pipeline from components (rather than using defaults) made it clear how sensitive topic granularity is to UMAP and HDBSCAN hyperparameters. Comparing three sentiment methods from different paradigms demonstrated that each captures a slightly different aspect of polarity, and that their disagreement is itself informative about ambiguous texts. Finally, the topic-level engagement analysis showed that thematic content is a meaningful predictor of post popularity, independent of sentiment.

---

## Tech Stack

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

## Summary of Results

The dataset comprises 476,842 Reddit records from approximately 3,000 unique users across three topic areas. The analysis confirmed that negative content attracts significantly higher engagement in AI-content and disinformation communities, while mental-health communities show the opposite pattern. Semantic embedding and UMAP projection revealed clear topical structure in the data, with mental-health discourse forming the most internally coherent cluster. BERTopic identified 29 micro-topics within the corpus, and temporal spike detection combined with LLM summarisation provided an efficient way to surface the events and discussions driving sudden shifts in community emotion.