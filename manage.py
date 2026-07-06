#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

# ── 强制离线模式：必须在所有 HuggingFace 相关导入之前设置 ──
# 如果 .env 中已配置，这里确保在进程级别生效
os.environ.setdefault("HF_HUB_OFFLINE", "1")

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
