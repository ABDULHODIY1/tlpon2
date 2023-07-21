from django.db import models
from django.urls import reverse
from django.conf import settings
# Create your models here.
User = settings.AUTH_USER_MODEL
class Cinema(models.Model):
    title = models.CharField(max_length=200)
    text = models.TextField()
    img = models.ImageField()
    narx = models.CharField(max_length=150)
    date = models.DateField(auto_now_add=True)
    likes = models.ManyToManyField(User, related_name='postlikes')
    slug = models.SlugField()
    author = models.ForeignKey(User, on_delete=models.CASCADE ,related_name='author')
    # user = models.OneToOneField(User, on_delete=models.CASCADE)


    def get_absolute_url(self):
        return reverse('detail',kwargs={"pk": self.pk})