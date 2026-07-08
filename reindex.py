"""重新索引所有已上传文档。

使用新的 chunk_size=1200、chunk_overlap=150、DOCX/PDF 标题检测策略。
"""
import os
import sys
import sqlite3
from pathlib import Path

# ── 必须在导入任何 HF/transformers 之前加载 .env ──
from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)
# 确保 HF_HOME 生效
print(f"HF_HOME={os.environ.get('HF_HOME', 'NOT SET')}")

# 设置 Django 环境
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
import django
django.setup()

from src.rag.indexer import Indexer

UPLOADS_DIR = Path(__file__).parent / "uploads"
INDEXES_DIR = Path(__file__).parent / "indexes"

# 文件 → 领域映射（从旧索引中读取）
DOMAIN_MAP = {}
old_db = INDEXES_DIR / "indexes.db"
if old_db.exists():
    conn = sqlite3.connect(str(old_db))
    rows = conn.execute("SELECT file_name, domain FROM files").fetchall()
    for name, domain in rows:
        DOMAIN_MAP[name] = domain
    conn.close()

print(f"旧索引领域映射: {DOMAIN_MAP}")

# 清除旧索引文件
print("\n=== 清除旧索引 ===")
for fname in ["indexes.db", "bm25.pkl", "faiss.index", "faiss_meta.json"]:
    fpath = INDEXES_DIR / fname
    if fpath.exists():
        fpath.unlink()
        print(f"  已删除: {fname}")

# D 方案：同时清理 Qdrant 本地存储，确保全量重建
qdrant_storage = INDEXES_DIR / "qdrant_storage"
if qdrant_storage.exists():
    import shutil
    shutil.rmtree(qdrant_storage)
    print("  已删除: qdrant_storage (Qdrant 向量库)")

# 创建新索引器
print("\n=== 创建新索引器 ===")
indexer = Indexer(
    uploads_dir=str(UPLOADS_DIR),
    indexes_dir=str(INDEXES_DIR),
)

# 列出所有上传文件
files = sorted([f for f in UPLOADS_DIR.iterdir() if f.is_file() and f.suffix.lower() in {".md", ".pdf", ".docx", ".txt"}])
print(f"待索引文件: {len(files)} 个\n")

# 逐个索引
success = 0
failed = 0
for f in files:
    domain = DOMAIN_MAP.get(f.name, "general")
    print(f"--- 索引: {f.name} (domain={domain}) ---")
    try:
        result = indexer.index_file(str(f), domain=domain)
        print(f"  ✅ {result['chunk_count']} 块\n")
        success += 1
    except Exception as e:
        print(f"  ❌ 失败: {e}\n")
        failed += 1

# 统计
print(f"\n=== 完成 ===")
print(f"成功: {success}, 失败: {failed}")
stats = indexer.get_stats()
print(f"总文件数: {stats['file_count']}")
print(f"总块数: {stats['chunk_count']}")
print(f"BM25 文档数: {stats['bm25_docs']}")
print(f"Qdrant 稠密文档数: {stats['dense_docs']}")
