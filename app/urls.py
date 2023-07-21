from .views import *
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static



urlpatterns = [
    path("",Home.as_view(),name='home'),
    path("crtldi/",MyCreateClass.as_view() ,name="CreatPost"),
    path("edit/<int:pk>/edit",UpdateView1.as_view(),name="yangilash"),
    path("detail/<int:pk>/",DetailModel.as_view(),name="detail"),
    path("delete/<int:pk>/",Deleteview.as_view(),name="Delete"),
    path("chat/UZAI",UZAI_Response,name="UZAI"),
    path("blogs/",Mydef,name="blogs")

]