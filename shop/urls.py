from django.urls import *
from .views import*

urlpatterns=[
    path("",shopshow.as_view(),name="shop1"),
]