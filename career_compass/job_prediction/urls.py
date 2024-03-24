from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_job, name='predict_job'),
    path('login/', views.user_login, name='user_login'),
    path('register/', views.user_register, name='user_register'),
]
