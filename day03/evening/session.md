To view stored session data in Django, you can access it in your views or templates. Here's a detailed guide on how to see the stored session data:

### Step-by-Step Guide

1. **Set Up Your Django Project**

   Ensure you have a Django project and app set up.

2. **Store Data in Session**

   First, let's make sure we're storing some data in the session. For this example, we'll use a form to store the user's name in the session.

   **views.py:**
   ```python
   from django.shortcuts import render, redirect
   from django import forms

   class MyForm(forms.Form):
       name = forms.CharField(max_length=100)

   def my_form_view(request):
       if request.method == 'POST':
           form = MyForm(request.POST)
           if form.is_valid():
               request.session['name'] = form.cleaned_data['name']
               return redirect('success_view')
       else:
           form = MyForm()

       return render(request, 'myapp/my_form.html', {'form': form})

   def success_view(request):
       return render(request, 'myapp/success.html')
   ```

3. **Create Templates**

   **my_form.html:**
   ```html
   <!DOCTYPE html>
   <html>
   <head>
       <title>My Form</title>
   </head>
   <body>
       <h1>My Form</h1>
       <form method="post">
           {% csrf_token %}
           {{ form.as_p }}
           <button type="submit">Submit</button>
       </form>
   </body>
   </html>
   ```

   **success.html:**
   ```html
   <!DOCTYPE html>
   <html>
   <head>
       <title>Success</title>
   </head>
   <body>
       <h1>Form Submitted Successfully</h1>
       <p>Welcome, {{ request.session.name }}!</p>
   </body>
   </html>
   ```

4. **Define URLs**

   **urls.py:**
   ```python
   from django.urls import path
   from .views import my_form_view, success_view

   urlpatterns = [
       path('form/', my_form_view, name='my_form'),
       path('success/', success_view, name='success_view'),
   ]
   ```

5. **View Session Data in Template**

   In the `success.html` template, we are already accessing the session data using `{{ request.session.name }}`.

6. **View Session Data in View**

   To see the session data in a view, you can simply access the `request.session` dictionary. For example, to print the session data to the console:

   **views.py:**
   ```python
   def success_view(request):
       name = request.session.get('name', 'Guest')
       print(f"Session data: {request.session.items()}")
       return render(request, 'myapp/success.html', {'name': name})
   ```

7. **Run the Server and Test**

    Run the Django development server:
    ```bash
    python manage.py runserver
    ```

    Visit `http://127.0.0.1:8000/form/` to fill out the form and submit it. You should be redirected to the success page, which displays the session data.

### Example: Accessing All Session Data in a View

If you want to access and display all session data in a view, you can iterate through the session dictionary and pass it to the template:

**views.py:**
```python
def success_view(request):
    session_data = request.session.items()
    return render(request, 'myapp/success.html', {'session_data': session_data})
```

**success.html:**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Success</title>
</head>
<body>
    <h1>Form Submitted Successfully</h1>
    <p>Welcome, {{ request.session.name }}!</p>

    <h2>Session Data:</h2>
    <ul>
        {% for key, value in session_data %}
            <li>{{ key }}: {{ value }}</li>
        {% endfor %}
    </ul>
</body>
</html>
```

This way, you can see all the session data stored in your Django application.

### Summary

By following these steps, you can store data in a Django session and access it in both views and templates. This is useful for maintaining user-specific data across different requests and pages.