from django.contrib.auth.forms import UserCreationForm,UserChangeForm
from .models import *

class CustomUserCreateForm(UserCreationForm):
     class Meta(UserCreationForm):
        model = CustomUser
        fields = ('username','first_name','email','age')

class CustomUserChanger(UserChangeForm):
    class Meta:
        model = CustomUser
        fields =('username','first_name','email','age')