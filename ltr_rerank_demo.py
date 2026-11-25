import json
import numpy as np
import xgboost as xgb
from opensearchpy import OpenSearch, exceptions

INDEX_NAME = "rec_products"
FEATURESET_NAME = "rec_features"
MODEL_NAME = "rec_xgb_model"

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}], use_ssl=False)


def init_ltr_store() -> None:
    """Initialize the .ltrstore index (idempotent)."""
    try:
        client.transport.perform_request("PUT", "/_ltr")
        print("LTR store (.ltrstore) initialized successfully.")
    except exceptions.RequestError as e:
        if "resource_already_exists_exception" in str(e):
            print("LTR store (.ltrstore) already exists, skipping initialization.")
        else:
            raise


def create_featureset() -> None:
    """Clean up old model/featureset and register the new featureset rec_features."""
    for path in [
        f"/_ltr/_model/{MODEL_NAME}",
        f"/_ltr/_featureset/{FEATURESET_NAME}",
    ]:
        try:
            client.transport.perform_request("DELETE", path)
        except exceptions.NotFoundError:
            pass

    featureset_body = {
        "featureset": {
            "features": [
                {
                    "name": "title_bm25",
                    "params": ["preferred_tags"],
                    "template_language": "mustache",
                    "template": {"match": {"title": "{{preferred_tags}}"}},
                },
                {
                    "name": "tags_bm25",
                    "params": ["preferred_tags"],
                    "template_language": "mustache",
                    "template": {"match": {"tags": "{{preferred_tags}}"}},
                },
                {
                    "name": "popularity_sat",
                    "params": [],
                    "template_language": "mustache",
                    "template": {
                        "rank_feature": {
                            "field": "popularity_score",
                            "saturation": {"pivot": 85.0},
                        }
                    },
                },
                {
                    "name": "category_match",
                    "params": ["target_category"],
                    "template_language": "mustache",
                    "template": {"term": {"category": "{{target_category}}"}},
                },
            ]
        }
    }
    client.transport.perform_request(
        "POST", f"/_ltr/_featureset/{FEATURESET_NAME}", body=featureset_body
    )
    print(f"Featureset [{FEATURESET_NAME}] registered successfully.")


def log_features_for_judgments() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use ext.ltr_log to extract the feature matrix for labeled judgment samples from OpenSearch."""
    # Mock judgment set: (qid, preferred_tags, target_category, {item_id: relevance grade 0~3})
    judgment_sessions = [
        (
            1,
            "noise-canceling wireless flagship",
            "audio",
            {"p001": 3, "p002": 3, "p010": 2, "p003": 1, "p005": 1, "p009": 0},
        ),
        (
            2,
            "outdoor camping portable",
            "outdoor",
            {"p007": 3, "p008": 3, "p010": 2, "p002": 1, "p006": 0, "p009": 0},
        ),
    ]

    X_rows, y_rows, qid_rows = [], [], []

    for qid, tags, target_cat, item_grades in judgment_sessions:
        query_body = {
            "size": len(item_grades),
            "_source": ["item_id"],
            "query": {
                "bool": {
                    "filter": [
                        {"ids": {"values": list(item_grades.keys())}},
                        {
                            "sltr": {
                                "_name": "logged_rec_features",
                                "featureset": FEATURESET_NAME,
                                "params": {
                                    "preferred_tags": tags,
                                    "target_category": target_cat,
                                },
                            }
                        },
                    ]
                }
            },
            "ext": {
                "ltr_log": {
                    "log_specs": {
                        "name": "log_entry",
                        "named_query": "logged_rec_features",
                        "missing_as_zero": True,
                    }
                }
            },
        }
        resp = client.search(index=INDEX_NAME, body=query_body)
        for hit in resp["hits"]["hits"]:
            item_id = hit["_source"]["item_id"]
            log_entry = hit["fields"]["_ltrlog"][0]["log_entry"]
            feat_vec = [entry.get("value", 0.0) for entry in log_entry]
            X_rows.append(feat_vec)
            y_rows.append(item_grades[item_id])
            qid_rows.append(qid)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    qid_arr = np.array(qid_rows, dtype=np.int32)
    print(f"Feature logging complete: {len(X)} samples, feature dimension {X.shape[1]}.")
    return X, y, qid_arr


def train_and_upload_xgboost(
    X: np.ndarray, y: np.ndarray, qid: np.ndarray
) -> None:
    """Train an XGBRanker ranking model and upload it to the OpenSearch LTR store."""
    ranker = xgb.XGBRanker(
        objective="rank:ndcg",
        n_estimators=12,
        max_depth=3,
        learning_rate=0.15,
        random_state=42,
    )
    ranker.fit(X, y, qid=qid)

    booster = ranker.get_booster()
    trees_json = [json.loads(tree) for tree in booster.get_dump(dump_format="json")]

    model_payload = {
        "model": {
            "name": MODEL_NAME,
            "model": {
                "type": "model/xgboost+json",
                "definition": {
                    "objective": "rank:ndcg",
                    "splits": trees_json,
                },
            },
        }
    }
    client.transport.perform_request(
        "POST",
        f"/_ltr/_featureset/{FEATURESET_NAME}/_createmodel",
        body=model_payload,
    )
    print(f"XGBoost model [{MODEL_NAME}] trained and uploaded successfully ({len(trees_json)} trees).")


def search_with_ltr_rescore(
    preferred_tags: str,
    target_category: str,
    purchased_ids: list[str],
    top_k: int = 5,
) -> None:
    """Stage 1 lightweight retrieval + Stage 2 LTR XGBoost rescore reranking."""
    query_body = {
        "size": top_k,
        "_source": ["item_id", "title", "category", "price"],
        "query": {
            "bool": {
                "filter": [{"term": {"in_stock": True}}],
                "must_not": [{"ids": {"values": purchased_ids}}],
                "should": [
                    {"match": {"tags": preferred_tags}},
                    {
                        "rank_feature": {
                            "field": "popularity_score",
                            "saturation": {"pivot": 85.0},
                        }
                    },
                ],
            }
        },
        "rescore": {
            "window_size": 100,
            "query": {
                "rescore_query": {
                    "sltr": {
                        "params": {
                            "preferred_tags": preferred_tags,
                            "target_category": target_category,
                        },
                        "model": MODEL_NAME,
                    }
                },
                "query_weight": 0.0,
                "rescore_query_weight": 1.0,
            },
        },
    }

    resp = client.search(index=INDEX_NAME, body=query_body)
    print(f"\n=== LTR Rescore Reranked Recommendations (Tags: '{preferred_tags}', Target Category: {target_category}) ===")
    for rank, hit in enumerate(resp["hits"]["hits"], start=1):
        src = hit["_source"]
        print(
            f"{rank}. [{src['item_id']}] {src['title']} "
            f"| Category: {src['category']} | Price: CNY {src['price']} | LTR Score: {hit['_score']:.4f}"
        )


if __name__ == "__main__":
    if not client.indices.exists(index=INDEX_NAME):
        raise RuntimeError("Please run `python rec_demo.py` first to initialize the rec_products index.")

    init_ltr_store()
    create_featureset()
    X_mat, y_vec, qids = log_features_for_judgments()
    train_and_upload_xgboost(X_mat, y_vec, qids)

    search_with_ltr_rescore(
        preferred_tags="noise-canceling wireless flagship",
        target_category="audio",
        purchased_ids=["p001"],
        top_k=5,
    )
