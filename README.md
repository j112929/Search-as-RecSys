# Search-as-RecSys (OpenSearch Demo)

A minimal runnable **Search-as-RecSys** demo built on **OpenSearch 2.19**, demonstrating how to perform multi-channel retrieval, hard filtering, and score fusion within a single search engine:

- **Tag / Keyword Retrieval (BM25)**: Matches user historical preference tags
- **Popularity Feature Boosting (`rank_feature`)**: Incorporates item popularity scores via a saturation function
- **Vector Semantic / Two-Tower Retrieval (`knn_vector` + HNSW)**: Performs nearest-neighbor retrieval based on user interest vectors
- **Hybrid Search Pipeline (`search_pipeline`)**: Applies `min_max` normalization across retrieval channels and combines scores using weighted arithmetic mean
- **Learning to Rank (LTR) Reranking**: Uses `sltr`, `ext.ltr_log`, and an XGBoost model to rerank candidates during the `rescore` phase
- **Business Rule Filtering**: Supports stock availability filtering (`in_stock: true`) and purchased-item deduplication (`must_not`)

## Quick Start

1. Start a single-node OpenSearch instance:

```bash
docker compose up -d
```

2. Install Python dependencies and run the basic hybrid recommendation demo:

```bash
pip install -r requirements.txt
python rec_demo.py
```

3. Run the PyTorch Two-Tower model training, partial vector sync, and real-time recommendation demo (run `rec_demo.py` first to initialize the index):

```bash
python two_tower_demo.py
```

4. Run the OpenSearch LTR feature logging, XGBoost training, and `rescore` reranking demo (run `rec_demo.py` first to initialize the index):

```bash
python ltr_rerank_demo.py
```
