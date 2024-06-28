Django models are Python classes that define the structure of your database tables. They serve as the blueprint for creating, reading, updating, and deleting records in the database. Here's a guide on how to work with models, database schema, and migrations in Django.

### 1. Define Models

In Django, you define your database schema using models. Each model corresponds to a single database table.

Open `myapp/models.py` and define your models:

```python
from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()

class Book(models.Model):
    title = models.CharField(max_length=200)
    publication_date = models.DateField()
    author = models.ForeignKey(Author, on_delete=models.CASCADE)
```

- `models.CharField`: A string field, for small-to-medium-sized strings.
- `models.EmailField`: A string field for email addresses.
- `models.DateField`: A date field.
- `models.ForeignKey`: A many-to-one relationship.

### 2. Database Schema

When you define models, Django creates the necessary database schema for you. The schema is generated based on the fields you define in your models.

### 3. Migrations

Migrations are how Django tracks changes to your models and applies them to your database schema. When you create or modify models, you need to create and apply migrations.

#### Create Migrations

Whenever you make changes to your models, create a new migration:

```bash
python manage.py makemigrations
```

Django will generate a migration file in the `migrations` directory of your app. For example, if you add a new field to the `Book` model, Django will create a migration file that reflects that change.

#### Apply Migrations

To apply the migrations and update your database schema, run:

```bash
python manage.py migrate
```

This command applies all pending migrations to the database.

### 4. Using Models

With your models defined and your database schema set up, you can start using your models to interact with the database.

#### Creating Records

```python
from myapp.models import Author, Book

# Create an author
author = Author(name="J.K. Rowling", email="jk@example.com")
author.save()

# Create a book
book = Book(title="Harry Potter and the Philosopher's Stone", publication_date="1997-06-26", author=author)
book.save()
```

#### Querying Records

```python
# Get all authors
authors = Author.objects.all()

# Get a specific author by ID
author = Author.objects.get(id=1)

# Filter authors by name
authors = Author.objects.filter(name="J.K. Rowling")

# Get all books by an author
books = Book.objects.filter(author=author)
```

#### Updating Records

```python
# Update an author's email
author.email = "new_email@example.com"
author.save()
```

#### Deleting Records

```python
# Delete a book
book.delete()
```

### 5. Admin Interface

Django comes with a built-in admin interface that allows you to manage your models through a web interface.

#### Register Models with Admin

Open `myapp/admin.py` and register your models:

```python
from django.contrib import admin
from .models import Author, Book

admin.site.register(Author)
admin.site.register(Book)
```

Now you can log in to the Django admin interface and manage your models.

### 6. Database Configuration

By default, Django uses SQLite as the database. You can change the database by modifying the `DATABASES` setting in `myproject/settings.py`:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',  # or 'django.db.backends.mysql', 'django.db.backends.sqlite3', etc.
        'NAME': 'mydatabase',
        'USER': 'mydatabaseuser',
        'PASSWORD': 'mypassword',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

Make sure you have the appropriate database driver installed (e.g., `psycopg2` for PostgreSQL).

### Summary

1. **Define Models**: Create models in `myapp/models.py`.
2. **Create Migrations**: `python manage.py makemigrations`
3. **Apply Migrations**: `python manage.py migrate`
4. **Use Models**: Interact with the database using the ORM.
5. **Admin Interface**: Register models in `admin.py` for admin access.
6. **Database Configuration**: Modify `DATABASES` setting in `settings.py` for your database of choice.

This guide covers the basics of working with models, database schema, and migrations in Django. With these tools, you can define your database structure and interact with your data effectively.