"""Belief Base: Knowledge management with vector retrieval.

Stores documents, facts, and environment perception data.
Supports ChromaDB for vector search with SQLite fallback.
"""

import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import config


class BeliefBase:
    """Knowledge base implementing the Belief component of BDI architecture.

    Beliefs represent the agent's knowledge about the world, including:
    - Static knowledge (documents, facts)
    - Dynamic perceptions (environment changes, observations)
    - Retrieved context for decision-making

    Supports two backends:
    - ChromaDB: vector search with semantic similarity
    - SQLite: fallback with keyword-based retrieval
    """

    def __init__(self, persist_dir: Optional[str] = None):
        """Initialize the belief base.

        Args:
            persist_dir: Path for persistent storage. Defaults to config value.
        """
        self._persist_dir = persist_dir or config.chroma.persist_dir
        self._backend: str = "sqlite"  # Default to sqlite, try chroma on demand
        self._chroma_collection: Any = None
        self._sqlite_conn: Optional[sqlite3.Connection] = None
        self._ttl: int = config.bdi.belief_ttl
        self._init_sqlite()

    # ------------------------------------------------------------------
    # Backend Initialization
    # ------------------------------------------------------------------

    def _init_sqlite(self) -> None:
        """Initialize SQLite fallback storage."""
        db_path = Path(self._persist_dir) / "beliefs.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._sqlite_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._sqlite_conn.execute("""
            CREATE TABLE IF NOT EXISTS beliefs (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                category TEXT DEFAULT 'general',
                created_at REAL NOT NULL,
                ttl REAL
            )
        """)
        self._sqlite_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_beliefs_category ON beliefs(category)
        """)
        self._sqlite_conn.commit()

    def _init_chroma(self) -> bool:
        """Try to initialize ChromaDB. Returns True on success."""
        if self._chroma_collection is not None:
            return True
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            chroma_path = Path(self._persist_dir) / "chroma"
            client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._chroma_collection = client.get_or_create_collection(
                name="beliefs",
                metadata={"hnsw:space": "cosine"},
            )
            self._backend = "chromadb"
            return True
        except Exception:
            self._chroma_collection = None
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_knowledge(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        category: str = "general",
        belief_id: Optional[str] = None,
    ) -> str:
        """Add a belief / knowledge entry.

        Args:
            content: The knowledge text content.
            metadata: Optional structured metadata.
            category: Classification tag for filtering.
            belief_id: Optional custom ID; auto-generated if not provided.

        Returns:
            The belief entry ID.
        """
        belief_id = belief_id or str(uuid.uuid4())
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        created_at = time.time()
        ttl_value = created_at + self._ttl if self._ttl > 0 else None

        # Always store in SQLite (reliable base)
        self._sqlite_conn.execute(
            "INSERT OR REPLACE INTO beliefs (id, content, metadata, category, created_at, ttl) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (belief_id, content, meta_json, category, created_at, ttl_value),
        )
        self._sqlite_conn.commit()

        # Also store in ChromaDB if available
        if self._init_chroma():
            try:
                self._chroma_collection.upsert(
                    ids=[belief_id],
                    documents=[content],
                    metadatas=[{"category": category, **(metadata or {})}],
                )
            except Exception:
                pass  # Chroma insert is best-effort

        return belief_id

    def query(
        self,
        query_text: str,
        category: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Query the belief base for relevant knowledge.

        Args:
            query_text: The search query.
            category: Optional category filter.
            top_k: Maximum number of results.

        Returns:
            List of matching belief entries with id, content, metadata, score.
        """
        results: List[Dict[str, Any]] = []

        # Try ChromaDB first for semantic search
        if self._init_chroma():
            try:
                where_filter = None
                if category:
                    where_filter = {"category": category}
                chroma_results = self._chroma_collection.query(
                    query_texts=[query_text],
                    n_results=top_k,
                    where=where_filter,
                )
                for i, doc_id in enumerate(chroma_results.get("ids", [[]])[0]):
                    results.append({
                        "id": doc_id,
                        "content": chroma_results["documents"][0][i],
                        "metadata": chroma_results.get("metadatas", [[]])[0][i] or {},
                        "score": float(1.0 - chroma_results.get("distances", [[0]])[0][i]),
                    })
                if results:
                    return results
            except Exception:
                pass  # Fall back to SQLite

        # SQLite fallback with keyword matching
        cursor = self._sqlite_conn.cursor()
        like_pattern = f"%{query_text}%"
        if category:
            cursor.execute(
                "SELECT id, content, metadata, category, created_at FROM beliefs "
                "WHERE (content LIKE ? OR category LIKE ?) AND category = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (like_pattern, like_pattern, category, top_k),
            )
        else:
            cursor.execute(
                "SELECT id, content, metadata, category, created_at FROM beliefs "
                "WHERE content LIKE ? OR category LIKE ? "
                "ORDER BY created_at DESC LIMIT ?",
                (like_pattern, like_pattern, top_k),
            )
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "content": row[1],
                "metadata": json.loads(row[2]) if row[2] else {},
                "category": row[3],
                "score": 0.5,  # Keyword match default score
            })
        return results

    def get_context(self, query_text: str, top_k: int = 5) -> str:
        """Get concatenated context string for LLM prompting.

        Args:
            query_text: The search query.
            top_k: Maximum number of results.

        Returns:
            Formatted context string, or empty string if nothing found.
        """
        results = self.query(query_text, top_k=top_k)
        if not results:
            return ""
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[{i}] {r['content']}")
        return "\n".join(parts)

    def get_by_id(self, belief_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific belief by ID.

        Args:
            belief_id: The belief entry ID.

        Returns:
            Dict with belief data or None if not found.
        """
        cursor = self._sqlite_conn.cursor()
        cursor.execute(
            "SELECT id, content, metadata, category, created_at FROM beliefs WHERE id = ?",
            (belief_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "content": row[1],
            "metadata": json.loads(row[2]) if row[2] else {},
            "category": row[3],
        }

    def list_by_category(self, category: str = "general") -> List[Dict[str, Any]]:
        """List all beliefs in a given category.

        Args:
            category: The category to filter by.

        Returns:
            List of belief entries.
        """
        cursor = self._sqlite_conn.cursor()
        cursor.execute(
            "SELECT id, content, metadata, category, created_at FROM beliefs "
            "WHERE category = ? ORDER BY created_at DESC",
            (category,),
        )
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "content": row[1],
                "metadata": json.loads(row[2]) if row[2] else {},
                "category": row[3],
            })
        return results

    def delete(self, belief_id: str) -> bool:
        """Delete a belief entry.

        Args:
            belief_id: The belief entry ID.

        Returns:
            True if deleted, False if not found.
        """
        cursor = self._sqlite_conn.cursor()
        cursor.execute("DELETE FROM beliefs WHERE id = ?", (belief_id,))
        self._sqlite_conn.commit()
        return cursor.rowcount > 0

    def clear_category(self, category: str) -> int:
        """Clear all beliefs in a given category.

        Args:
            category: The category to clear.

        Returns:
            Number of entries removed.
        """
        cursor = self._sqlite_conn.cursor()
        cursor.execute("DELETE FROM beliefs WHERE category = ?", (category,))
        self._sqlite_conn.commit()
        return cursor.rowcount

    def count(self) -> int:
        """Return total number of stored beliefs."""
        cursor = self._sqlite_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM beliefs")
        return cursor.fetchone()[0]
