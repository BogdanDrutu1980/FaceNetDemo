"""
Definition of views.
"""

from datetime import datetime
from django.shortcuts import render
from django.http import HttpRequest

from app.models import Face
from django.contrib.staticfiles.storage import staticfiles_storage
from django.core.files.storage import FileSystemStorage

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
    if request.method == 'POST':
        #form = UploadFileForm(request.POST, request.FILES)
        #if form.is_valid():
            #form.save()

        uploaded_file = request.FILES['document']
        print(uploaded_file.name)
        fs = FileSystemStorage()
        fs.save(uploaded_file.name, uploaded_file)
        f = Face()

        return render(
            request,
            'app/predict.html',
            {
                'title':'About',
                'mess':f.getName(),
                'year':datetime.now().year,
            }
        )


    return render(request, 'app/predict.html', {}
    )
