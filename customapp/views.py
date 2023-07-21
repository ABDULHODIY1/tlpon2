from django.shortcuts import render
from django.views.generic import CreateView,TemplateView,ListView
from django.contrib.auth.mixins import LoginRequiredMixin,UserPassesTestMixin
from .forms import CustomUserCreateForm
from app.models import Cinema
from django.urls import reverse_lazy
from django.conf import settings
# Create your views here.
User = settings.AUTH_USER_MODEL
class SingUpView(CreateView):
    form_class = CustomUserCreateForm
    success_url = reverse_lazy('login')
    template_name = 'registration/singup.html'



class UserHome(TemplateView):
    # models= Cinema
    template_name = 'users/users.html'
    # def test_func(self):
    #     obj = self.get_object()
    #     obj.author == User


    