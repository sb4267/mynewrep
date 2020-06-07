from django.db import models

class Snippet(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    img = models.ImageField(upload_to='images/')
    class Meta:
        ordering = ['created']
