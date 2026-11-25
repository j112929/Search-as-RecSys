import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from opensearchpy import OpenSearch, helpers

# 复用 rec_demo.py 中的索引与管道配置
INDEX_NAME = "rec_products"
PIPELINE_NAME = "rec-hybrid-pipeline"
EMBED_DIM = 8

# 1. 商品基础元数据（用于 Item 塔融合类目特征解决冷启动）
ITEMS = [
    {"item_id": "p001", "category": "audio", "title": "Sony WH-1000XM5 降噪耳机"},
    {"item_id": "p002", "category": "audio", "title": "Bose QuietComfort Ultra 耳塞"},
    {"item_id": "p003", "category": "audio", "title": "国产性价比降噪蓝牙耳机 AirLite"},
    {"item_id": "p004", "category": "audio", "title": "缺货限量版 Hi-Fi 监听耳机"},
    {"item_id": "p005", "category": "electronics", "title": "Apple MacBook Air M3 笔记本"},
    {"item_id": "p006", "category": "electronics", "title": "Keychron K2 Pro 无线机械键盘"},
    {"item_id": "p007", "category": "outdoor", "title": "始祖鸟 Beta 防水冲锋衣"},
    {"item_id": "p008", "category": "outdoor", "title": "牧高笛露营折叠蛋卷桌椅套装"},
    {"item_id": "p009", "category": "home", "title": "Fellow Stagg EKG 手冲咖啡壶"},
    {"item_id": "p010", "category": "audio", "title": "Shokz 韶音 OpenRun Pro 运动耳机"},
]

CAT2IDX = {"audio": 0, "electronics": 1, "outdoor": 2, "home": 3}
SEG2IDX = {"tech_geek": 0, "outdoor_lover": 1, "home_aesthetic": 2}
ITEM2IDX = {it["item_id"]: idx for idx, it in enumerate(ITEMS)}
PAD_IDX = len(ITEMS)

# 2. 模拟用户交互样本：(用户分群, 历史点击序列, 正样本目标商品)
RAW_INTERACTIONS = [
    ("tech_geek", ["p001", "p005"], "p002"),
    ("tech_geek", ["p002", "p006"], "p001"),
    ("tech_geek", ["p001", "p002"], "p005"),
    ("tech_geek", ["p005", "p006"], "p003"),
    ("tech_geek", ["p001", "p010"], "p002"),
    ("outdoor_lover", ["p007", "p008"], "p010"),
    ("outdoor_lover", ["p007", "p010"], "p008"),
    ("outdoor_lover", ["p008"], "p007"),
    ("home_aesthetic", ["p006", "p009"], "p005"),
    ("home_aesthetic", ["p009"], "p006"),
    ("home_aesthetic", ["p008", "p009"], "p006"),
    ("home_aesthetic", ["p005", "p009"], "p006"),
]


class RecDataset(Dataset):
    def __init__(self, interactions, max_hist_len: int = 3):
        self.samples = []
        for seg, hist_ids, target_id in interactions:
            hist_idx = [ITEM2IDX[i] for i in hist_ids][-max_hist_len:]
            hist_idx = [PAD_IDX] * (max_hist_len - len(hist_idx)) + hist_idx
            target_idx = ITEM2IDX[target_id]
            target_cat = CAT2IDX[ITEMS[target_idx]["category"]]
            self.samples.append(
                (
                    torch.tensor(SEG2IDX[seg], dtype=torch.long),
                    torch.tensor(hist_idx, dtype=torch.long),
                    torch.tensor(target_idx, dtype=torch.long),
                    torch.tensor(target_cat, dtype=torch.long),
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# 3. 双塔模型定义
class UserTower(nn.Module):
    def __init__(self, num_segments: int, num_items: int, embed_dim: int):
        super().__init__()
        self.pad_idx = num_items
        self.seg_emb = nn.Embedding(num_segments, 16)
        self.hist_item_emb = nn.Embedding(num_items + 1, 16, padding_idx=self.pad_idx)
        self.mlp = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
        )

    def forward(self, seg_idx: torch.Tensor, hist_items: torch.Tensor) -> torch.Tensor:
        seg_vec = self.seg_emb(seg_idx)  # [B, 16]
        mask = (hist_items != self.pad_idx).unsqueeze(-1).float()  # [B, L, 1]
        hist_vec = (self.hist_item_emb(hist_items) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        x = torch.cat([seg_vec, hist_vec], dim=-1)  # [B, 32]
        return F.normalize(self.mlp(x), p=2, dim=-1)


class ItemTower(nn.Module):
    def __init__(self, num_items: int, num_cats: int, embed_dim: int):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, 16)
        self.cat_emb = nn.Embedding(num_cats, 16)
        self.mlp = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
        )

    def forward(self, item_idx: torch.Tensor, cat_idx: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.item_emb(item_idx), self.cat_emb(cat_idx)], dim=-1)  # [B, 32]
        return F.normalize(self.mlp(x), p=2, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, num_segments: int, num_items: int, num_cats: int, embed_dim: int):
        super().__init__()
        self.user_tower = UserTower(num_segments, num_items, embed_dim)
        self.item_tower = ItemTower(num_items, num_cats, embed_dim)

    def compute_infonce_loss(
        self,
        seg_idx: torch.Tensor,
        hist_items: torch.Tensor,
        pos_item_idx: torch.Tensor,
        pos_cat_idx: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        user_vecs = self.user_tower(seg_idx, hist_items)         # [B, D]
        item_vecs = self.item_tower(pos_item_idx, pos_cat_idx)   # [B, D]
        logits = torch.matmul(user_vecs, item_vecs.T) / temperature  # [B, B]
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)


def train_two_tower(epochs: int = 120) -> TwoTowerModel:
    torch.manual_seed(42)
    dataset = RecDataset(RAW_INTERACTIONS)
    loader = DataLoader(dataset, batch_size=6, shuffle=True, drop_last=True)

    model = TwoTowerModel(
        num_segments=len(SEG2IDX),
        num_items=len(ITEMS),
        num_cats=len(CAT2IDX),
        embed_dim=EMBED_DIM,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for seg_idx, hist_items, pos_item, pos_cat in loader:
            optimizer.zero_grad()
            loss = model.compute_infonce_loss(seg_idx, hist_items, pos_item, pos_cat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 30 == 0:
            print(f"Epoch {epoch:3d}/{epochs} | InfoNCE Loss: {total_loss / len(loader):.4f}")

    return model


def export_and_sync_items(model: TwoTowerModel, client: OpenSearch) -> None:
    """用 Item Tower 导出全量商品向量，并局部更新写入 OpenSearch"""
    model.eval()
    item_indices = torch.arange(len(ITEMS), dtype=torch.long)
    cat_indices = torch.tensor([CAT2IDX[it["category"]] for it in ITEMS], dtype=torch.long)

    with torch.no_grad():
        item_vecs = model.item_tower(item_indices, cat_indices).cpu().numpy().tolist()

    actions = [
        {
            "_op_type": "update",
            "_index": INDEX_NAME,
            "_id": ITEMS[i]["item_id"],
            "doc": {"item_vector": item_vecs[i]},
            "doc_as_upsert": False,
        }
        for i in range(len(ITEMS))
    ]
    success, errors = helpers.bulk(client, actions, refresh=True, raise_on_error=False)
    print(f"已将双塔 Item 向量同步到 OpenSearch：成功 {success} 条，失败 {len(errors)} 条。")


def query_with_user_tower(
    model: TwoTowerModel,
    client: OpenSearch,
    seg_name: str,
    history_ids: list[str],
    preferred_tags: str,
    top_k: int = 5,
) -> None:
    """用 User Tower 实时推理生成 user_vector，并执行 OpenSearch 混合推荐查询"""
    model.eval()
    hist_idx = [ITEM2IDX[i] for i in history_ids][-3:]
    hist_idx = [PAD_IDX] * (3 - len(hist_idx)) + hist_idx

    with torch.no_grad():
        user_vec = (
            model.user_tower(
                torch.tensor([SEG2IDX[seg_name]], dtype=torch.long),
                torch.tensor([hist_idx], dtype=torch.long),
            )
            .squeeze(0)
            .cpu()
            .numpy()
            .tolist()
        )

    query_body = {
        "size": top_k,
        "_source": ["item_id", "title", "category", "price"],
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "bool": {
                            "filter": [{"term": {"in_stock": True}}],
                            "must_not": [{"ids": {"values": history_ids}}],
                            "should": [{"match": {"tags": preferred_tags}}],
                        }
                    },
                    {
                        "knn": {
                            "item_vector": {
                                "vector": user_vec,
                                "k": 10,
                                "filter": {
                                    "bool": {
                                        "filter": [{"term": {"in_stock": True}}],
                                        "must_not": [{"ids": {"values": history_ids}}],
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
    print(f"\n=== 双塔实时推荐（分群: {seg_name}, 历史: {history_ids}）===")
    for rank, hit in enumerate(resp["hits"]["hits"], start=1):
        src = hit["_source"]
        print(f"{rank}. [{src['item_id']}] {src['title']} | 类目: {src['category']} | 得分: {hit['_score']:.4f}")


if __name__ == "__main__":
    os_client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}], use_ssl=False)
    if not os_client.indices.exists(index=INDEX_NAME):
        raise RuntimeError("请先运行 python rec_demo.py 初始化索引与 search_pipeline。")

    trained_model = train_two_tower(epochs=120)
    export_and_sync_items(trained_model, os_client)

    query_with_user_tower(
        model=trained_model,
        client=os_client,
        seg_name="tech_geek",
        history_ids=["p001", "p005"],
        preferred_tags="降噪 无线 旗舰",
    )
