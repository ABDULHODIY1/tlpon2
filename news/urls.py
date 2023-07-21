from django.urls import *
from .views import *

urlpatterns=[
    path("news", News.as_view(),name="News"),
]