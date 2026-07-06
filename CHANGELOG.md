---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: b31ef81ac61f9e32d6ed53bf7ba2c350_b3e5daf6755c11f1a7da5254006c9bbf
    ReservedCode1: AuGWap0t3D76YB1vlXwzpfe1RwGzDVDBDVvSfmSZUfTgD9HYKSiRsrzsBA2dxxaDYMZRQrXr7YsPF7k57Lr50UlqkZ6L/m9W6nEB005h17wa2e0pwEyfCPuv/IS9L3hGkgfhRrrc5fYOUDaaXS3hTzxZiHLn2cF3C55KntWAXqS2JhDfOskIEwH8NTk=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: b31ef81ac61f9e32d6ed53bf7ba2c350_b3e5daf6755c11f1a7da5254006c9bbf
    ReservedCode2: AuGWap0t3D76YB1vlXwzpfe1RwGzDVDBDVvSfmSZUfTgD9HYKSiRsrzsBA2dxxaDYMZRQrXr7YsPF7k57Lr50UlqkZ6L/m9W6nEB005h17wa2e0pwEyfCPuv/IS9L3hGkgfhRrrc5fYOUDaaXS3hTzxZiHLn2cF3C55KntWAXqS2JhDfOskIEwH8NTk=
---

# 变更记录

## 2026-07-01 — 语义缓存: 从 SimHash 两级匹配 → FAISS 一步匹配 + 源文件新鲜度校验

### 背景

原实现采用 SimHash 语义指纹做汉明预筛，再走 FAISS 余弦精确匹配的两级架构。经过评估，SimHash 层存在以下问题：

1. 指纹二值化丢失语义精度，预筛增加延迟而无实质收益
2. 两级架构不属于业界主流（主流方案：embedding → FAISS 直接匹配）
3. 缺少源文件新鲜度校验——源文件更新后旧缓存仍可能命中，返回过期内容

### 改动

**1. 源文件新鲜度校验**（先做）

- `src/semantic_cache.py` 新增 `_source_files`（`|` 分隔的多文件路径）和 `_indexed_at`（最大 mtime）字段
- `search()` 余弦命中后调用 `_is_entry_fresh(idx)`：逐文件比对存在性 + mtime，任一变更即过期
- 过期自动 `_invalidate_entry(idx)`：从 FAISS 索引和所有列表中剔除
- `add()` 签名扩展：`add(question, answer, source_file="", indexed_at=0.0)`
- `dump()` / `_try_load()` 完整持久化新字段

- `src/workflows/orchestrator.py` 新增 `_collect_source_info(retrieved_context)`
  - 从 `doc_id`（格式 `filename::hash::chunk_idx`）解析源文件路径
  - 返回 `(source_files_str, max_mtime)`
- `research_node` 写缓存时注入源文件信息

**2. 移除 SimHash，切换业界主流架构**（后做）

- `src/semantic_cache.py` 删除全部 SimHash 相关代码：
  - 常量：`_FINGERPRINT_BITS`, `_HAMMING_THRESHOLD`, `_PREFILTER_TOPK`
  - 工具函数：`_pack_bits`, `_unpack_bits`, `_hamming_distance`, `_hamming_distances`
  - 字段：`_fingerprints`, `_projections`, `_bytes_per_fp`
  - 方法：`_init_projections`, `compute_fingerprint`, `fingerprint_hex`, `search_by_fingerprint`
  - 持久化：`fingerprints.npy`, `projections.npy`
- `search()` 从「SimHash 预筛 → FAISS 精确匹配」改为「FAISS IndexFlatIP.search(k=1) 一步到位」
- `add()` 去重从指纹比对改为 FAISS 余弦匹配（更准确）
- `__init__` 精简参数：移除 `n_bits`, `hamming_threshold`, `prefilter_topk`
- `stats()` 移除 `n_bits`, `hamming_threshold`, `projections_loaded`

### 影响

- 对外接口完全兼容：`search()`, `add()`, `dump()`, `_try_load()` 调用方无需修改
- 旧持久化文件（`fingerprints.npy`, `projections.npy`）不再需要，下次启动会自动忽略
- 缓存命中延迟降低（少一层 SimHash 计算）
- 零额外 LLM token 消耗（全程只用 embedding 模型）
*（内容由AI生成，仅供参考）*
