"""
Definition of models.
"""

from django.db import models
from django.contrib.staticfiles.storage import staticfiles_storage

from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from os import listdir
from matplotlib import pyplot
from os.path import isdir
import numpy as np
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model
from random import choice
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import os
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
        faces = list()
        y=list()
        for filename in listdir(directory):
            path = directory+ '/'+ filename
            face = self.getFace(path)
            faces.append(face)

        dir=os.path.basename(os.path.normpath(directory))
        y = [dir for _ in range(len(faces))]
        self.faces=asarray(faces)
        self.labels=asarray(y)
        self.facesLoaded=True
        return self.faces,self.labels


    def getFace(self,filename):
        # load image from file
        image = Image.open(filename)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = asarray(image)
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(self.requiredSize)
        faceArray = asarray(image)
        return faceArray

    def getEmbeddings(self,face_pixels):
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = self.model.predict(samples)
        return yhat[0]

    def getSetEmbeddings(self):
        if self.facesLoaded:
            newTrainX = list()
            for face_pixels in self.faces:
                embedding = self.getEmbeddings(face_pixels)
                newTrainX.append(embedding)
            newTrainX = asarray(newTrainX)
            return newTrainX
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

    def loadDataset(self, filename):
        # load face embeddings
        data = load(filename)
        self.filename=filename
        self.trainX, self.trainy, self.testX, self.testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

        self.dsLoaded = True
    def updateDataset(self,directory):
        if self.dsLoaded and self.embeddingsModelLoaded:
            fa = FaceEmbeddings(self.modelEmbeddings)
            new_x_train,new_y_train=fa.getFaces(directory)
            new_x_train=fa.getSetEmbeddings()

            new_x_train.reshape(-1, 1)
            new_y_train.reshape(-1, 1)

            self.trainX = np.concatenate((self.trainX,new_x_train), axis=0)
            self.trainy = np.concatenate((self.trainy,new_y_train), axis=0)
            savez_compressed(self.filename,self.trainX, self.trainy, self.testX, self.testy)


    def loadEmbeddingsModel(self, model):
        # load the facenet model
        self.modelEmbeddings = load_model(model)
        print('Embeddings model loaded!')
        self.embeddingsModelLoaded = True

    def fit(self):
        if self.dsLoaded:
            # normalize input vectors
            in_encoder = Normalizer(norm='l2')
            self.trainX = in_encoder.transform(self.trainX)
            self.testX = in_encoder.transform(self.testX)
            # label encode targets
            self.out_encoder = LabelEncoder()
            self.out_encoder.fit(self.trainy)
            self.trainy = self.out_encoder.transform(self.trainy)
            self.testy = self.out_encoder.transform(self.testy)
            #model.fit
            self.modelSVC = SVC(kernel='linear', probability=True)
            self.modelSVC.fit(self.trainX, self.trainy)
            self.modelFitted = True
            print("SVC model fitted!!!")
        else:
            print("Dataset not loaded!!!")

    def predict(self, filename):
        if self.dsLoaded and self.embeddingsModelLoaded and self.modelFitted:
            print("Conditions to predict OK!!!")
            fe = FaceEmbeddings(self.modelEmbeddings)
            random_face_pixels = fe.getFace(filename)
            random_face_emb = fe.getEmbeddings(random_face_pixels)

            samples = expand_dims(random_face_emb, axis=0)
            yhat_class = self.modelSVC.predict(samples)
            yhat_prob = self.modelSVC.predict_proba(samples)
            # get name
            class_index = yhat_class[0]
            class_probability = yhat_prob[0,class_index] * 100
            predict_names = self.out_encoder.inverse_transform(yhat_class)
            return predict_names[0], class_probability
            #print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
            # plot for fun
            #pyplot.imshow(random_face_pixels)
            #title = '%s (%.3f)' % (predict_names[0], class_probability)
            #pyplot.title(title)
            #pyplot.show()
        else:
            print("Conditions not OK!!!")

class Face(models.Model):
    def __init__(self):
        rec = Recognizer()
        rec.loadEmbeddingsModel(os.path.join(settings.STATIC_ROOT, 'facenet_keras.h5'))
        #rec.loadDataset(os.path.join(settings.STATIC_ROOT, '5-celebrity-faces-embeddings.npz'))
        #rec.fit()
        #self.name, self.probability = rec.predict(os.path.join(settings.MEDIA_ROOT, 'documents/test.png'))
        self.name = "test"
    
    def getName(self):
        return self.name
    def handle_uploaded_file(f):
        pass
