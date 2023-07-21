from django.db import models

# Create your models here.
class NewsModel(models.Model):
    title=models.CharField(max_length=200)
    text=models.TextField()
    image=models.ImageField(blank=True, null=True)
    p=models.IntegerField()