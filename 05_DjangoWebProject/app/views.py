"""
Definition of views.
"""

from datetime import datetime
from django.shortcuts import render
from django.http import HttpRequest

from app.models import Face
from django.contrib.staticfiles.storage import staticfiles_storage
from django.core.files.storage import FileSystemStorage

fa = None

def home(request):
    """Renders the home page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/index.html',
        {
            'title':'Home Page',
            'year':datetime.now().year,
        }
    )
def init(request):
    """Renders the init page (button)."""
    assert isinstance(request, HttpRequest)
    global fa
    fa = Face()
    return render(
        request,
        'app/init.html',
        {
            'title':'Init  Page',
            'year':datetime.now().year,
        }
    )



def update(request):
    """Renders the contact page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/update.html',
        {
            'title':'Update',
            'message':'Your update page.',
            'year':datetime.now().year,
        }
    )

def about(request):
    """Renders the about page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/about.html',
        {
            'title':'About',
            'message':'Your application description page.',
            'year':datetime.now().year,
        }
    )

def predict(request):
    """Renders the about page."""
    assert isinstance(request, HttpRequest)
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        print(uploaded_file.name)
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        context['url'] = fs.url(name)
        return render(
            request,
            'app/predict.html',
            context
        )


    return render(request, 'app/predict.html', {}
    )
