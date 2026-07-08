from django.apps import AppConfig


class CoreConfig(AppConfig):
    name = 'core'

    def ready(self):
        from django.conf import settings
        host = getattr(settings, 'RUNSERVER_HOST', '127.0.0.1')
        port = getattr(settings, 'RUNSERVER_PORT', '8000')
        print()
        print('  Django Admin:  http://%s:%s/' % (host, port))
        print('  Vue Frontend:  http://localhost:5173')
        print()

        # 初始化结构化文件日志（每日轮转，保留 30 天）
        # setup_logging 会 clear handlers，之后需重新挂载 LogStoreHandler 保证内存日志
        try:
            import logging
            from src.core.logging_config import setup_logging
            setup_logging()
            # 重新挂载内存日志 handler（供前端 /api/system/logs 查询）
            from src.log_store import log_store, LogStoreHandler
            _mem_handler = LogStoreHandler()
            _mem_handler.setLevel(logging.INFO)
            _mem_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            logging.getLogger().addHandler(_mem_handler)
            logging.getLogger("MASApp").info("文件日志 + 内存日志已初始化", extra={"component": "app"})
        except Exception as e:
            print(f"  ⚠️ 日志初始化失败: {e}")
