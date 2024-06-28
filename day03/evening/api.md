Django REST framework (DRF) is a powerful and flexible toolkit for building Web APIs. Here's a comprehensive guide to developing APIs with Django REST framework.

### 1. Installation

First, install Django REST framework.

```bash
pip install djangorestframework
```

Add `'rest_framework'` to your `INSTALLED_APPS` in `settings.py`.

```python
# settings.py
INSTALLED_APPS = [
    ...
    'rest_framework',
    ...
]
```

### 2. Setting Up Your Project

Create a new Django project and app.

```bash
django-admin startproject myproject
cd myproject
python manage.py startapp myapp
```

### 3. Models

Define your models in `models.py`.

```python
# myapp/models.py
from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=100)
    author = models.CharField(max_length=100)
    published_date = models.DateField()

    def __str__(self):
        return self.title
```

Run migrations to create the database tables.

```bash
python manage.py makemigrations
python manage.py migrate
```

### 4. Serializers

Serializers define how model instances are converted to JSON and vice versa.

```python
# myapp/serializers.py
from rest_framework import serializers
from .models import Book

class BookSerializer(serializers.ModelSerializer):
    class Meta:
        model = Book
        fields = '__all__'
```

### 5. Views

DRF provides several view classes for handling API requests. The most common ones are `APIView`, `GenericAPIView`, and viewsets.

#### Using APIView

```python
# myapp/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Book
from .serializers import BookSerializer

class BookList(APIView):
    def get(self, request):
        books = Book.objects.all()
        serializer = BookSerializer(books, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = BookSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
```

#### Using ViewSets

ViewSets combine logic for multiple views, reducing boilerplate code.

```python
# myapp/views.py
from rest_framework import viewsets
from .models import Book
from .serializers import BookSerializer

class BookViewSet(viewsets.ModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
```

### 6. URLs

Wire up the views with URLs. DRF provides a router class to automatically generate URL patterns for viewsets.

```python
# myapp/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import BookViewSet

router = DefaultRouter()
router.register(r'books', BookViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
```

Include the app URLs in the project’s `urls.py`.

```python
# myproject/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('myapp.urls')),
]
```

### 7. Authentication and Permissions

DRF provides several built-in authentication and permission classes.

#### Authentication

Add authentication classes to your settings.

```python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.BasicAuthentication',
    ],
}
```

#### Permissions

Define permission classes in your views.

```python
# myapp/views.py
from rest_framework.permissions import IsAuthenticated

class BookViewSet(viewsets.ModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
    permission_classes = [IsAuthenticated]
```

### 8. Pagination

DRF supports pagination out of the box. Configure pagination in your settings.

```python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 10,
}
```

### 9. Filtering and Ordering

Use Django filters to filter and order API results.

#### Install Django Filter

```bash
pip install django-filter
```

Add it to your `INSTALLED_APPS`.

```python
# settings.py
INSTALLED_APPS = [
    ...
    'django_filters',
]
```

Configure DRF to use Django Filter.

```python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend',
    ],
}
```

Define filters in your views.

```python
# myapp/views.py
from rest_framework import viewsets
from django_filters.rest_framework import DjangoFilterBackend
from .models import Book
from .serializers import BookSerializer

class BookViewSet(viewsets.ModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['author', 'published_date']
```

### 10. Testing

Write tests for your API using Django’s test framework.

```python
# myapp/tests.py
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from .models import Book

class BookTests(APITestCase):
    def test_create_book(self):
        url = reverse('book-list')
        data = {'title': 'Test Book', 'author': 'Author', 'published_date': '2024-01-01'}
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Book.objects.count(), 1)
        self.assertEqual(Book.objects.get().title, 'Test Book')
```

Run your tests with:

```bash
python manage.py test
```

### Summary

1. **Installation**: Install Django REST framework and add it to `INSTALLED_APPS`.
2. **Models**: Define your models in `models.py`.
3. **Serializers**: Create serializers to convert model instances to JSON.
4. **Views**: Use `APIView` or viewsets to handle API requests.
5. **URLs**: Use routers to generate URL patterns for viewsets.
6. **Authentication and Permissions**: Implement authentication and permissions.
7. **Pagination**: Configure pagination to manage large querysets.
8. **Filtering and Ordering**: Use Django Filter to filter and order API results.
9. **Testing**: Write tests to ensure your API works correctly.

This guide covers the basics of building an API with Django REST framework, enabling you to create, read, update, and delete resources through a RESTful API.