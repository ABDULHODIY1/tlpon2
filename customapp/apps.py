from django.apps import AppConfig


class CustomappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'customapp'
