In Django, views handle the logic for your web application, and templates define the HTML structure. Together, they form the basis of the front-end of your application. Here’s a guide on how to work with views and templates in Django.

### 1. Views

Views are Python functions or classes that receive a web request and return a web response. They typically interact with your models and render templates.

#### Function-based Views

In `myapp/views.py`, create a simple view:

```python
from django.shortcuts import render
from .models import Book

def index(request):
    books = Book.objects.all()
    return render(request, 'myapp/index.html', {'books': books})
```

- `render(request, template_name, context)`: A shortcut to render a template with a context dictionary.

#### Class-based Views

Django also supports class-based views which offer more structure and reusability.

```python
from django.views.generic import ListView
from .models import Book

class BookListView(ListView):
    model = Book
    template_name = 'myapp/index.html'
    context_object_name = 'books'
```

- `ListView`: A generic view for displaying a list of objects.
- `model`: The model to query.
- `template_name`: The template to render.
- `context_object_name`: The name of the context variable to use in the template.

### 2. Templates

Templates are HTML files that define the structure and layout of your web pages. They can include placeholders for dynamic content, which are populated by the context passed from views.

#### Template Directory

Create a directory named `templates` inside your app directory and then create a subdirectory with the same name as your app. For example:

```
myapp/
    templates/
        myapp/
            index.html
```

#### Template Syntax

Django uses a template language to define placeholders and control logic in templates.

##### Variables

To include dynamic content, use double curly braces:

```html
<!-- myapp/templates/myapp/index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Book List</title>
</head>
<body>
    <h1>Book List</h1>
    <ul>
        {% for book in books %}
            <li>{{ book.title }} by {{ book.author.name }}</li>
        {% endfor %}
    </ul>
</body>
</html>
```

- `{{ book.title }}`: Inserts the title of the book.
- `{% for book in books %} ... {% endfor %}`: Loops over the books and generates list items for each one.

##### Filters

You can apply filters to variables to modify their output.

```html
{{ book.title|upper }}
```

- `|upper`: Converts the book title to uppercase.

##### Tags

Django template tags control the logic of your templates.

```html
{% if books %}
    <ul>
        {% for book in books %}
            <li>{{ book.title }}</li>
        {% endfor %}
    </ul>
{% else %}
    <p>No books available.</p>
{% endif %}
```

- `{% if %} ... {% else %} ... {% endif %}`: Conditional statement to check if there are books.

#### Template Inheritance

Django supports template inheritance to avoid repeating common elements like headers and footers.

```html
<!-- myapp/templates/myapp/base.html -->
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}My Site{% endblock %}</title>
</head>
<body>
    <header>
        <h1>Welcome to My Site</h1>
    </header>
    <main>
        {% block content %}{% endblock %}
    </main>
</body>
</html>
```

```html
<!-- myapp/templates/myapp/index.html -->
{% extends 'myapp/base.html' %}

{% block title %}Book List{% endblock %}

{% block content %}
    <h1>Book List</h1>
    <ul>
        {% for book in books %}
            <li>{{ book.title }} by {{ book.author.name }}</li>
        {% endfor %}
    </ul>
{% endblock %}
```

- `{% extends 'myapp/base.html' %}`: Indicates that this template extends `base.html`.
- `{% block %} ... {% endblock %}`: Defines blocks that can be overridden by child templates.

### 3. Rendering Templates in Views

You’ve already seen how to use `render()` in function-based views to render a template. For class-based views, Django automatically handles rendering if you specify the `template_name` attribute.

### 4. URL Configuration

Ensure your views are connected to URLs so that they can be accessed via web requests. Open `myapp/urls.py` and configure your URLs:

```python
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # For function-based view
    # path('', views.BookListView.as_view(), name='index'),  # For class-based view
]
```

Include your app’s URLs in the project’s `urls.py`:

```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('myapp/', include('myapp.urls')),
]
```

### Summary

1. **Views**: Create function-based or class-based views in `myapp/views.py`.
2. **Templates**: Create HTML files in `myapp/templates/myapp/`.
3. **Template Syntax**: Use Django template language for variables, filters, and tags.
4. **Template Inheritance**: Use `{% extends %}` and `{% block %}` for reusable layouts.
5. **Rendering**: Use `render()` in views to render templates with context.
6. **URL Configuration**: Map views to URLs in `myapp/urls.py`.

This guide covers the basics of working with views and templates in Django, enabling you to build dynamic and maintainable web pages.