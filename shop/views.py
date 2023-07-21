from django.shortcuts import render
from django.views.generic import*
from .models import*

# Create your views here.

class shopshow(ListView):
    model=Shop
    template_name="shop/shop1.html"
    context_object_name="shop"