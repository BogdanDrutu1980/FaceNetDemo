"""
Definition of models.
"""

from django.db import models
from django.contrib.staticfiles.storage import staticfiles_storage

from random import choice
import os
from PIL import Image
from django.conf import settings
# Create your models here.


class FaceEmbeddings:
    def __init__(self, model):
        self.requiredSize = (160,160)
        self.model = model
        self.faces=None
        self.labels=None
        self.facesLoaded=False

    def getFaces(self,directory):
        return self.faces,self.labels
    
    
    def getFace(self,filename):
        # load image from file
        image = Image.open(filename)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = asarray(image)
        faceArray = asarray(image)
        return faceArray
    
class Recognizer:
    def __init__(self):
        self.dsLoaded = False
        self.embeddingsModelLoaded = False
        self.svcModelLoaded=False
        self.modelFitted = False
        self.trainX = None
        self.trainy = None
        self.testX = None
        self.texty = None
        self.filename=None
        self.out_encoder = None
    
            
    def predict(self, filename):
        print("Conditions not OK!!!")

class Face(models.Model):
    def __init__(self):
        rec = Recognizer()
        #rec.loadEmbeddingsModel(os.path.join(settings.STATIC_ROOT, 'facenet_keras.h5'))
        #rec.loadDataset(os.path.join(settings.STATIC_ROOT, '5-celebrity-faces-embeddings.npz'))
        #rec.fit()
        #self.name, self.probability = rec.predict(os.path.join(settings.MEDIA_ROOT, 'documents/test.png'))
    def getName(self):
        return self.name
    def handle_uploaded_file(f):
        pass 
