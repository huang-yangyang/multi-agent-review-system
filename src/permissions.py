"""权限管理 — 企业级文档权限 + 租户隔离 + 工具白名单。

设计原则：权限逻辑在中间层，不交给 LLM 判断。
"""

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional

from src.config import config
from src.core.logging_config import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# 数据库初始化
# ═══════════════════════════════════════════════════════════════

_DB_PATH = Path(config.rag.indexes_dir) / "permissions.db"
_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db():
    # 迁移：补 domain 列
    try:
        conn = _get_conn()
        conn.execute("SELECT domain FROM doc_permissions LIMIT 1")
        conn.close()
    except Exception:
        conn = _get_conn()
        conn.execute("ALTER TABLE doc_permissions ADD COLUMN domain TEXT DEFAULT ''")
        conn.commit()
        conn.close()
    with _lock:
        conn = _get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS doc_permissions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_path    TEXT NOT NULL,
                doc_name    TEXT NOT NULL,
                owner       TEXT NOT NULL,
                tenant_id   TEXT NOT NULL DEFAULT 'default',
                permission  TEXT NOT NULL DEFAULT 'admin',
                domain      TEXT DEFAULT '',
                created_at  TEXT DEFAULT (datetime('now')),
                UNIQUE(doc_path, owner)
            );
            CREATE INDEX IF NOT EXISTS idx_dp_tenant ON doc_permissions(tenant_id);
            CREATE INDEX IF NOT EXISTS idx_dp_owner ON doc_permissions(owner);
            CREATE INDEX IF NOT EXISTS idx_dp_domain ON doc_permissions(domain);

            CREATE TABLE IF NOT EXISTS role_tools (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                role        TEXT NOT NULL,
                tool_name   TEXT NOT NULL,
                granted     INTEGER DEFAULT 1,
                UNIQUE(role, tool_name)
            );

            CREATE TABLE IF NOT EXISTS temp_grants (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                owner       TEXT NOT NULL,
                resource    TEXT NOT NULL,
                permission  TEXT NOT NULL DEFAULT 'read',
                granted_by  TEXT NOT NULL,
                expires_at  TEXT NOT NULL,
                used        INTEGER DEFAULT 0,
                created_at  TEXT DEFAULT (datetime('now'))
            );
        """)
        conn.commit()
        conn.close()


_init_db()
# 迁移：补 domain 列
try:
    conn = _get_conn()
    conn.execute("SELECT domain FROM doc_permissions LIMIT 1")
    conn.close()
except Exception:
    conn = _get_conn()
    conn.execute("ALTER TABLE doc_permissions ADD COLUMN domain TEXT DEFAULT ''")
    conn.commit()
    conn.close()

# 迁移：补 visibility 列（统一可见性模型），并回填旧 permission/domain 数据
try:
    conn = _get_conn()
    conn.execute("SELECT visibility FROM doc_permissions LIMIT 1")
    conn.close()
except Exception:
    conn = _get_conn()
    conn.execute("ALTER TABLE doc_permissions ADD COLUMN visibility TEXT DEFAULT 'admin'")
    rows = conn.execute("SELECT id, permission, domain FROM doc_permissions").fetchall()
    for rid, perm, dom in rows:
        if perm == "read":
            vis = "public"
        elif dom == "law":
            vis = "legal"
        elif dom == "general":
            vis = "hr"
        else:
            vis = "admin"
        conn.execute("UPDATE doc_permissions SET visibility=? WHERE id=?", (vis, rid))
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════
# 租户标识
# ═══════════════════════════════════════════════════════════════

def get_tenant_id(request) -> str:
    """从请求中提取租户 ID（当前用 username 作为租户标识）。"""
    if hasattr(request, 'user_name'):
        return request.user_name
    return "anonymous"


# ═══════════════════════════════════════════════════════════════
# 文档权限（统一可见性模型）
# ═══════════════════════════════════════════════════════════════

# 可见性取值 → 可访问角色集合
#   admin      仅系统管理员
#   legal_lead 仅法律主管
#   hr_lead    仅人事主管
#   legal      法律主管 + 法律员工
#   hr         人事主管 + 人事员工
#   public     所有用户（含员工）
VISIBILITY_ROLES = {
    "admin": {"admin"},
    "legal_lead": {"legal_lead"},
    "hr_lead": {"hr_lead"},
    "legal": {"legal_lead", "legal_user"},
    "hr": {"hr_lead", "hr_user"},
    "public": set(),  # 空集合表示「所有人」
}
VALID_VISIBILITIES = set(VISIBILITY_ROLES.keys())


def _visibility_to_permission(visibility: str) -> str:
    """兼容旧字段：public → read（公开），其余 → admin。"""
    return "read" if visibility == "public" else "admin"


def _visibility_to_domain(visibility: str) -> str:
    """兼容旧字段：推导领域，便于索引分类。"""
    if visibility in ("legal", "legal_lead"):
        return "law"
    if visibility in ("hr", "hr_lead"):
        return "general"
    return ""


def grant_doc_permission(doc_path: str, doc_name: str, owner: str, tenant_id: str = "default", visibility: str = "admin", domain: str = ""):
    """授予文档权限（上传时调用）。visibility 取值见 VALID_VISIBILITIES。"""
    if visibility not in VALID_VISIBILITIES:
        visibility = "admin"
    try:
        with _lock:
            conn = _get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO doc_permissions (doc_path, doc_name, owner, tenant_id, permission, domain, visibility) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_path, doc_name, owner, tenant_id, _visibility_to_permission(visibility), domain or _visibility_to_domain(visibility), visibility),
            )
            conn.commit()
            conn.close()
    except Exception as e:
        logger.warning(f"grant_doc_permission failed: {e}")


def set_doc_visibility(doc_path: str, visibility: str, tenant_id: str = "default"):
    """修改文档可见性，保留原 owner。visibility 取值见 VALID_VISIBILITIES。"""
    if visibility not in VALID_VISIBILITIES:
        visibility = "admin"
    try:
        with _lock:
            conn = _get_conn()
            cur = conn.execute(
                "UPDATE doc_permissions SET visibility=?, permission=?, domain=? WHERE doc_path=? AND tenant_id=?",
                (visibility, _visibility_to_permission(visibility), _visibility_to_domain(visibility), doc_path, tenant_id),
            )
            if cur.rowcount == 0:
                conn.execute(
                    "INSERT INTO doc_permissions (doc_path, doc_name, owner, tenant_id, permission, domain, visibility) VALUES (?,?,?,?,?,?,?)",
                    (doc_path, Path(doc_path).name, "anonymous", tenant_id, _visibility_to_permission(visibility), _visibility_to_domain(visibility), visibility),
                )
            conn.commit()
            conn.close()
    except Exception as e:
        logger.warning(f"set_doc_visibility failed: {e}")
def get_doc_visibility(doc_path: str, tenant_id: str = "default") -> str:
    """获取文档的可见性标签（admin/legal_lead/hr_lead/legal/hr/public）。"""
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT visibility FROM doc_permissions WHERE doc_path = ? AND tenant_id = ? LIMIT 1",
            (doc_path, tenant_id),
        ).fetchone()
        conn.close()
        if row and row[0]:
            return row[0]
    except Exception:
        pass
    return "admin"



def get_user_docs(owner: str, tenant_id: str = "default") -> List[dict]:
    """获取用户可访问的文档列表。返回 doc_path 列表。"""
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT doc_path, doc_name, visibility FROM doc_permissions WHERE tenant_id = ? AND (owner = ? OR visibility = 'public')",
            (tenant_id, owner),
        ).fetchall()
        conn.close()
        return [{"path": r[0], "name": r[1], "visibility": r[2]} for r in rows]
    except Exception as e:
        logger.warning(f"get_user_docs failed: {e}")
        return []


def user_can_access_doc(doc_path: str, owner: str, tenant_id: str = "default") -> bool:
    """检查用户是否有权访问某个文档（含角色可见性）。"""
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT visibility, owner FROM doc_permissions WHERE doc_path = ? AND tenant_id = ? LIMIT 1",
            (doc_path, tenant_id),
        ).fetchone()
        conn.close()
        if not row:
            return True  # 无记录 → 降级允许
        vis, doc_owner = row[0], row[1]
        if doc_owner == owner:
            return True
        if vis == "public":
            return True
        role = _get_user_role(owner)
        if role == "admin":
            return True
        return role in VISIBILITY_ROLES.get(vis, set())
    except Exception:
        return True  # 降级：故障时允许访问


def get_user_accessible_doc_paths(owner: str, tenant_id: str = "default") -> set:
    """获取用户有权访问的所有文档路径（按统一的 visibility 可见性模型）。

    返回值：
    - None → 不需要过滤（管理员或权限表为空，看全部）
    - 空集 set() → 用户没有任何可访问文档（应过滤掉所有结果）
    - 非空集 → 用户可访问的文档路径集合

    可见性规则（见 VISIBILITY_ROLES）：
    - admin      仅系统管理员
    - legal_lead 仅法律主管
    - hr_lead    仅人事主管
    - legal      法律主管 + 法律员工
    - hr         人事主管 + 人事员工
    - public     所有用户（含员工）

    此外：管理员看全部；自己上传的文档始终可见。权限表为空 → None（降级看全部）。
    """
    # 使用直接 SQL 查询而非 Django ORM，避免 async 上下文中的 SynchronousOnlyOperation
    try:
        db_path = Path(config.path.project_root) / "db.sqlite3"
        conn_django = sqlite3.connect(str(db_path))
        row = conn_django.execute(
            "SELECT up.role, u.is_superuser FROM core_userprofile up "
            "JOIN auth_user u ON up.user_id = u.id "
            "WHERE u.username = ? LIMIT 1",
            (owner,)
        ).fetchone()
        if not row:
            # 没有 UserProfile，检查是否是 superuser
            row = conn_django.execute(
                "SELECT is_superuser FROM auth_user WHERE username = ? LIMIT 1",
                (owner,)
            ).fetchone()
            conn_django.close()
            if row and row[0]:
                return None  # superuser → 看全部
            role = "general_user"
        else:
            conn_django.close()
            role, is_superuser = row[0], row[1]
            if role == "admin" or is_superuser:
                return None  # 管理员看全部
    except Exception:
        role = "general_user"

    try:
        conn = _get_conn()
        # 先检查权限表是否有数据
        total = conn.execute("SELECT COUNT(*) FROM doc_permissions").fetchone()[0]
        if total == 0:
            conn.close()
            return None  # 权限表空 → 看全部（首次使用降级）

        # 不再按 tenant_id 过滤：公开文档对所有用户可见
        # 统一按 visibility 判定：public=所有人；其余按 VISIBILITY_ROLES 角色集合；
        # 自己上传的文档始终可见。
        rows = conn.execute(
            "SELECT doc_path, visibility, owner FROM doc_permissions"
        ).fetchall()
        conn.close()
        allowed = set()
        for doc_path, vis, doc_owner in rows:
            # 仅对真实已认证用户生效「自己的文档自动可见」；
            # 未鉴权请求（owner='anonymous'）不继承匿名所有权，避免越权看到全部文档
            if doc_owner == owner and owner != "anonymous":
                allowed.add(doc_path)
                continue
            if vis == "public":
                allowed.add(doc_path)
                continue
            if role in VISIBILITY_ROLES.get(vis, set()):
                allowed.add(doc_path)
        return allowed
    except Exception:
        return None


def revoke_doc_permission(doc_path: str, owner: str):
    """撤销文档权限。"""
    try:
        with _lock:
            conn = _get_conn()
            conn.execute("DELETE FROM doc_permissions WHERE doc_path = ? AND owner = ?", (doc_path, owner))
            conn.commit()
            conn.close()
    except Exception as e:
        logger.warning(f"revoke_doc_permission failed: {e}")


def get_user_cache_scope(owner: str) -> str:
    """计算用户的缓存权限范围标识。

    用于语义缓存隔离：不同权限级别的用户使用不同 scope，
    确保管理员的缓存答案不会泄漏给普通用户。

    Returns:
        "admin" — 管理员（可看全部文档，所有管理员共享缓存空间）
        "role:<role>:<hash>" — 普通用户（相同角色+相同可访问文档集的用户共享缓存）
    """
    try:
        accessible = get_user_accessible_doc_paths(owner)
        if accessible is None:
            return "admin"
        if not accessible:
            return f"none:{owner}"  # 无权限用户，单独缓存空间
        role = _get_user_role(owner)
        # 对可访问文档路径排序后哈希，确保相同权限的用户共享缓存
        sorted_paths = sorted(accessible)
        scope_hash = hashlib.md5("|".join(sorted_paths).encode()).hexdigest()[:12]
        return f"role:{role}:{scope_hash}"
    except Exception:
        return f"fallback:{owner}"


# ═══════════════════════════════════════════════════════════════
# 工具白名单
# ═══════════════════════════════════════════════════════════════

_DEFAULT_TOOLS = {
    "admin": ["kb_search", "web_search", "calculate", "file_upload", "file_delete", "cache_manage"],
    "user":  ["kb_search", "web_search", "calculate", "file_upload"],
    "auditor": ["kb_search"],  # 只读
}


def init_default_tools():
    """初始化默认角色工具白名单。"""
    with _lock:
        conn = _get_conn()
        for role, tools in _DEFAULT_TOOLS.items():
            for tool in tools:
                conn.execute(
                    "INSERT OR IGNORE INTO role_tools (role, tool_name, granted) VALUES (?, ?, 1)",
                    (role, tool),
                )
        conn.commit()
        conn.close()


init_default_tools()


def user_can_use_tool(username: str, tool_name: str) -> bool:
    """检查用户是否有权使用某个工具。"""
    role = _get_user_role(username)
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT 1 FROM role_tools WHERE role = ? AND tool_name = ? AND granted = 1 LIMIT 1",
            (role, tool_name),
        ).fetchone()
        conn.close()
        return row is not None
    except Exception:
        return tool_name in _DEFAULT_TOOLS.get(role, [])


def _get_user_role(username: str) -> str:
    """获取用户角色（直接 SQL 查询，兼容 async 上下文）。"""
    try:
        db_path = Path(config.path.project_root) / "db.sqlite3"
        conn_django = sqlite3.connect(str(db_path))
        row = conn_django.execute(
            "SELECT up.role, u.is_superuser FROM core_userprofile up "
            "JOIN auth_user u ON up.user_id = u.id "
            "WHERE u.username = ? LIMIT 1",
            (username,)
        ).fetchone()
        if row:
            conn_django.close()
            role, is_superuser = row[0], row[1]
            if role == "admin" or is_superuser:
                return "admin"
            return role
        # 没有 UserProfile，检查是否是 superuser
        row = conn_django.execute(
            "SELECT is_superuser FROM auth_user WHERE username = ? LIMIT 1",
            (username,)
        ).fetchone()
        conn_django.close()
        if row and row[0]:
            return "admin"
    except Exception:
        pass
    return "general_user"


# ═══════════════════════════════════════════════════════════════
# 临时授权
# ═══════════════════════════════════════════════════════════════

def grant_temp_permission(owner: str, resource: str, permission: str, granted_by: str, expires_hours: int = 24):
    """授予临时权限。"""
    from datetime import datetime, timedelta
    expires = (datetime.now() + timedelta(hours=expires_hours)).isoformat()
    try:
        with _lock:
            conn = _get_conn()
            conn.execute(
                "INSERT INTO temp_grants (owner, resource, permission, granted_by, expires_at) VALUES (?, ?, ?, ?, ?)",
                (owner, resource, permission, granted_by, expires),
            )
            conn.commit()
            conn.close()
            logger.info(f"临时授权: {owner} → {resource} ({permission}) 有效期 {expires_hours}h")
    except Exception as e:
        logger.warning(f"grant_temp_permission failed: {e}")


def check_temp_permission(owner: str, resource: str) -> bool:
    """检查是否有有效的临时授权（自动清理过期）。"""
    from datetime import datetime
    try:
        conn = _get_conn()
        # 先清理过期
        conn.execute("DELETE FROM temp_grants WHERE expires_at < ?", (datetime.now().isoformat(),))
        conn.commit()
        row = conn.execute(
            "SELECT 1 FROM temp_grants WHERE owner = ? AND resource = ? AND expires_at > ? LIMIT 1",
            (owner, resource, datetime.now().isoformat()),
        ).fetchone()
        conn.close()
        return row is not None
    except Exception:
        return False
