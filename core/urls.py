"""URL routing for the core app — 14 endpoints migrated from FastAPI."""

from django.urls import path
from . import views

urlpatterns = [
    # Health checks
    path('health', views.health_check),
    path('api/health', views.health_check),
    path('api/metrics', views.metrics),
    path('api/graph', views.graph_structure),

    # Admin page
    path('', views.admin_page),

    # File upload & management
    path('api/upload', views.upload_file),
    path('api/files', views.list_files),
    path('api/files/<path:filename>', views.delete_file),

    # RAG
    path('api/rag/search', views.rag_search),

    # Agent endpoints
    path('api/workflow', views.workflow),
    path('api/workflow/resume', views.workflow_resume),

    # Dashboard
    path('api/dashboard', views.dashboard),

    # Auth
    path('api/auth/login', views.auth_login),
    path('api/auth/logout', views.auth_logout),

    # Conversations & Logs
    path('api/conversations', views.list_conversations),
    path('api/conversations/<str:conv_id>/messages', views.conversation_messages),
    path('api/conversations/<str:conv_id>', views.update_conversation_view),   # PATCH
    path('api/conversations/<str:conv_id>/delete', views.delete_conversation_view),  # DELETE (explicit path to avoid PATCH conflict)
    path('api/logs', views.list_logs),
    path('api/audit', views.audit_log_view),

    # Semantic Cache
    path('api/cache/list', views.cache_list),
    path('api/cache/entry/<int:entry_id>', views.cache_delete_entry),
    path('api/cache/clear', views.cache_clear_all),
    path('api/cache/domain/<str:domain>', views.cache_delete_by_domain),
]
