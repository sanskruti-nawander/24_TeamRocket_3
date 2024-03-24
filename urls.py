from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_job, name='predict_job'),
]
