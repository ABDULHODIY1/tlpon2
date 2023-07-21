from django.db import models

# Create your models here.
class Shop(models.Model):
    title=models.CharField(max_length=200)
    text=models.TextField()
    url=models.URLField()
    img=models.ImageField()