import time
import numpy as np
from opensearchpy import OpenSearch, helpers

INDEX_NAME = "rec_products"
PIPELINE_NAME = "rec-hybrid-pipeline"
VECTOR_DIM = 8

# 1. Connect to local OpenSearch
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    use_ssl=False,
    verify_certs=False,
)


def wait_for_opensearch(retries: int = 30) -> None:
    for _ in range(retries):
        try:
            if client.ping():
                print("OpenSearch is ready.")
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("Timed out connecting to OpenSearch. Please check Docker container status.")


def setup_search_pipeline() -> None:
    """Register hybrid search normalization and weighted fusion pipeline: 35% rules/text/popularity, 65% vector similarity."""
    pipeline_body = {
        "description": "Min-max normalization and weighted average fusion for multi-channel recommendation scores",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": [0.35, 0.65]},
                    },
                }
            }
        ],
    }
    client.transport.perform_request(
        "PUT", f"/_search/pipeline/{PIPELINE_NAME}", body=pipeline_body
    )
    print(f"Search Pipeline [{PIPELINE_NAME}] created successfully.")


def setup_index() -> None:
    """Create product index with text, category, popularity feature, and HNSW vector fields."""
    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)

    mapping = {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "properties": {
                "item_id": {"type": "keyword"},
                "title": {"type": "text"},
                "tags": {"type": "text"},
                "category": {"type": "keyword"},
                "brand": {"type": "keyword"},
                "price": {"type": "float"},
                "in_stock": {"type": "boolean"},
                "popularity_score": {"type": "rank_feature"},
                "item_vector": {
                    "type": "knn_vector",
                    "dimension": VECTOR_DIM,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                        "parameters": {"ef_construction": 128, "m": 16},
                    },
                },
            }
        },
    }
    client.indices.create(index=INDEX_NAME, body=mapping)
    print(f"Index [{INDEX_NAME}] created successfully.")


def normalize(vec: list[float]) -> list[float]:
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return (arr / norm).tolist() if norm > 0 else arr.tolist()


def ingest_mock_products() -> None:
    """Ingest 10 mock products (8-D vector: [tech, audio, outdoor, home, value, flagship, portable, aesthetic])."""
    raw_items = [
        {
            "item_id": "p001",
            "title": "Sony WH-1000XM5 Wireless Noise-Canceling Over-Ear Headphones",
            "tags": "noise-canceling wireless bluetooth headphones commute flagship audio",
            "category": "audio",
            "brand": "Sony",
            "price": 2499.0,
            "in_stock": True,
            "popularity_score": 95.0,
            "item_vector": normalize([0.9, 0.95, 0.1, 0.2, 0.2, 0.9, 0.7, 0.8]),
        },
        {
            "item_id": "p002",
            "title": "Bose QuietComfort Ultra Noise-Canceling Earbuds",
            "tags": "noise-canceling wireless bluetooth earbuds portable sports",
            "category": "audio",
            "brand": "Bose",
            "price": 1999.0,
            "in_stock": True,
            "popularity_score": 88.0,
            "item_vector": normalize([0.85, 0.9, 0.5, 0.1, 0.25, 0.85, 0.95, 0.7]),
        },
        {
            "item_id": "p003",
            "title": "AirLite Budget Wireless Noise-Canceling Bluetooth Earbuds",
            "tags": "noise-canceling wireless bluetooth headphones student value",
            "category": "audio",
            "brand": "SoundX",
            "price": 299.0,
            "in_stock": True,
            "popularity_score": 92.0,
            "item_vector": normalize([0.7, 0.6, 0.2, 0.2, 0.95, 0.1, 0.85, 0.6]),
        },
        {
            "item_id": "p004",
            "title": "Out-of-Stock Limited Edition Hi-Fi Studio Monitor Headphones",
            "tags": "Hi-Fi studio monitor wired audiophile",
            "category": "audio",
            "brand": "Sennheiser",
            "price": 3999.0,
            "in_stock": False,  # Test inventory filtering
            "popularity_score": 99.0,
            "item_vector": normalize([0.8, 0.99, 0.0, 0.3, 0.05, 0.98, 0.2, 0.6]),
        },
        {
            "item_id": "p005",
            "title": "Apple MacBook Air M3 Ultraportable Laptop",
            "tags": "thin-and-light office tech flagship portable",
            "category": "electronics",
            "brand": "Apple",
            "price": 8999.0,
            "in_stock": True,
            "popularity_score": 96.0,
            "item_vector": normalize([0.95, 0.3, 0.1, 0.4, 0.15, 0.95, 0.85, 0.95]),
        },
        {
            "item_id": "p006",
            "title": "Keychron K2 Pro Custom Wireless Mechanical Keyboard",
            "tags": "wireless mechanical keyboard office desk-setup aesthetic",
            "category": "electronics",
            "brand": "Keychron",
            "price": 549.0,
            "in_stock": True,
            "popularity_score": 84.0,
            "item_vector": normalize([0.85, 0.1, 0.0, 0.6, 0.75, 0.4, 0.5, 0.9]),
        },
        {
            "item_id": "p007",
            "title": "Arc'teryx Beta Lightweight Waterproof Hardshell Jacket",
            "tags": "outdoor hiking waterproof hardshell jacket lightweight",
            "category": "outdoor",
            "brand": "Arc'teryx",
            "price": 4500.0,
            "in_stock": True,
            "popularity_score": 90.0,
            "item_vector": normalize([0.1, 0.0, 0.98, 0.1, 0.1, 0.95, 0.8, 0.75]),
        },
        {
            "item_id": "p008",
            "title": "MobiGarden Foldable Camping Roll-Top Table & Chair Set",
            "tags": "camping outdoor portable home patio",
            "category": "outdoor",
            "brand": "MobiGarden",
            "price": 429.0,
            "in_stock": True,
            "popularity_score": 82.0,
            "item_vector": normalize([0.05, 0.0, 0.9, 0.65, 0.85, 0.2, 0.75, 0.7]),
        },
        {
            "item_id": "p009",
            "title": "Fellow Stagg EKG Temperature-Controlled Pour-Over Kettle",
            "tags": "home coffee desk-setup aesthetic minimalist design",
            "category": "home",
            "brand": "Fellow",
            "price": 1299.0,
            "in_stock": True,
            "popularity_score": 86.0,
            "item_vector": normalize([0.2, 0.0, 0.05, 0.95, 0.2, 0.8, 0.3, 0.98]),
        },
        {
            "item_id": "p010",
            "title": "Shokz OpenRun Pro Bone Conduction Sports Headphones",
            "tags": "sports running wireless bluetooth headphones outdoor",
            "category": "audio",
            "brand": "Shokz",
            "price": 1298.0,
            "in_stock": True,
            "popularity_score": 89.0,
            "item_vector": normalize([0.75, 0.65, 0.92, 0.1, 0.4, 0.7, 0.95, 0.75]),
        },
    ]

    actions = [
        {"_index": INDEX_NAME, "_id": item["item_id"], "_source": item}
        for item in raw_items
    ]
    helpers.bulk(client, actions, refresh=True)
    print(f"Successfully ingested {len(raw_items)} mock products.")


def recommend_for_user(
    user_name: str,
    preferred_tags: str,
    user_vector: list[float],
    purchased_ids: list[str],
    top_k: int = 5,
) -> None:
    """Run hybrid recommendation query: BM25 tags + popularity saturation boost + k-NN vector retrieval + inventory & deduplication filters."""
    norm_user_vec = normalize(user_vector)

    query_body = {
        "size": top_k,
        "_source": ["item_id", "title", "category", "brand", "price", "tags"],
        "query": {
            "hybrid": {
                "queries": [
                    # Channel 1: Tag/text match + popularity rank_feature saturation boost
                    {
                        "bool": {
                            "filter": [{"term": {"in_stock": True}}],
                            "must_not": [{"ids": {"values": purchased_ids}}],
                            "should": [
                                {"match": {"tags": {"query": preferred_tags, "boost": 1.5}}},
                                {
                                    "rank_feature": {
                                        "field": "popularity_score",
                                        "saturation": {"pivot": 85.0},
                                        "boost": 0.5,
                                    }
                                },
                            ],
                        }
                    },
                    # Channel 2: User interest vector k-NN retrieval (with identical inventory & purchased filters)
                    {
                        "knn": {
                            "item_vector": {
                                "vector": norm_user_vec,
                                "k": 10,
                                "filter": {
                                    "bool": {
                                        "filter": [{"term": {"in_stock": True}}],
                                        "must_not": [{"ids": {"values": purchased_ids}}],
                                    }
                                },
                            }
                        }
                    },
                ]
            }
        },
    }

    resp = client.search(
        index=INDEX_NAME,
        body=query_body,
        params={"search_pipeline": PIPELINE_NAME},
    )

    print(f"\n=== Top-{top_k} Hybrid Recommendations for User [{user_name}] ===")
    for rank, hit in enumerate(resp["hits"]["hits"], start=1):
        src = hit["_source"]
        print(
            f"{rank}. [{src['item_id']}] {src['title']} "
            f"| Category: {src['category']} | Price: CNY {src['price']} | Hybrid Score: {hit['_score']:.4f}"
        )


if __name__ == "__main__":
    wait_for_opensearch()
    setup_search_pipeline()
    setup_index()
    ingest_mock_products()

    # Scenario A: High-end tech & audio enthusiast (already bought p001, filter out to avoid duplicate recommendations)
    recommend_for_user(
        user_name="Tech & Audio Geek (Purchased Sony XM5)",
        preferred_tags="noise-canceling wireless flagship portable",
        user_vector=[0.9, 0.9, 0.1, 0.2, 0.1, 0.95, 0.85, 0.85],
        purchased_ids=["p001"],
        top_k=5,
    )

    # Scenario B: Outdoor camping & home aesthetics enthusiast
    recommend_for_user(
        user_name="Outdoor Camping & Home Lifestyle Enthusiast",
        preferred_tags="outdoor camping home portable",
        user_vector=[0.1, 0.1, 0.9, 0.85, 0.5, 0.6, 0.7, 0.9],
        purchased_ids=[],
        top_k=5,
    )
