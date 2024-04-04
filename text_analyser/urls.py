from django.urls import path
from .views import text_analyser_view

urlpatterns = [
    path('', text_analyser_view, name='text_analyser'),
]