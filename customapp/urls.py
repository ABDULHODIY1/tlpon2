from django.urls import path
from .views import *

urlpatterns = [
    path('singup/', SingUpView.as_view(), name='singup'),
    path('UserHome/',UserHome.as_view(),name='users'),
    ]