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
