# Search-as-RecSys (OpenSearch Demo)

基于 **OpenSearch 2.19** 实现的“基于搜索的推荐系统（Search-as-RecSys）”最小可运行 Demo，演示如何在同一搜索引擎中完成多路召回、硬过滤与分数融合：

- **标签 / 关键词召回（BM25）**：匹配用户历史偏好标签
- **热度特征加权（`rank_feature`）**：使用饱和函数引入商品热度分
- **向量语义 / 双塔召回（`knn_vector` + HNSW）**：基于 8 维用户兴趣向量做近邻召回
- **混合检索管道（`search_pipeline`）**：对两路分数进行 `min_max` 归一化并按权重加权融合
- **业务规则过滤**：支持库存状态（`in_stock: true`）与已购商品去重（`must_not`）

## 快速运行

1. 启动单节点 OpenSearch：

```bash
docker compose up -d
```

2. 安装 Python 依赖并运行推荐脚本：

```bash
pip install -r requirements.txt
python rec_demo.py
```
