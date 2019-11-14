#!/usr/bin/python

from pymongo import MongoClient
import tornado.web

from tornado.web import HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options

from basehandler import BaseHandler

from sklearn.neighbors import KNeighborsClassifier
import pickle
from bson.binary import Binary
import json
import numpy as np

import base64
import os
from PIL import Image
import cv2
# Take in base64 string and return cv image




def stringToRGB(base64_string):
    img = base64.b64decode(str(base64_string)); 
    npimg = np.fromstring(img, dtype=np.uint8); 
    source = cv2.imdecode(npimg, 1)
    source = cv2.resize(source,(300,400))
    #     # Filename 
    # filename = 'savedImage.jpg'
      
    # # Using cv2.imwrite() method 
    # # Saving the image 
    # cv2.imwrite(filename, source) 
    return source


class PrintHandlers(BaseHandler):
    def get(self):
        '''Write out to screen the handlers used
        This is a nice debugging example!
        '''
        self.set_header("Content-Type", "application/json")
        self.write(self.application.handlers_string.replace('),','),\n'))

class UploadLabeledDatapointHandler(BaseHandler):
    def post(self):
        '''Save data point and class label to database
        '''
        data = json.loads(self.request.body.decode("utf-8"))

        vals = data['feature']
        image = stringToRGB(vals)
        # vals = image
        print('\n\n\n\n',image.shape)
        # fvals = [float(val) for val in vals]
        label = data['label']
        sess  = data['dsid']

        dbid = self.db.labeledinstances.insert(
            {"feature":vals,"label":label,"dsid":sess}
            );
        self.write_json({"id":str(dbid),
            "feature":vals,
            "label":label})


        face = self.faceEmbedding(image)

        print('\n\n\n\n',face.shape)

        # # save image 
        # directory = self.image_dataset_dir + '/'+label
        # if not os.path.exists(directory):
        #     os.makedirs(directory)

        # img_number=len(os.listdir(directory))

class RequestNewDatasetId(BaseHandler):
    def get(self):
        '''Get a new dataset ID for building a new dataset
        '''
        a = self.db.labeledinstances.find_one(sort=[("dsid", -1)])
        if a == None:
            newSessionId = 1
        else:
            newSessionId = float(a['dsid'])+1
        self.write_json({"dsid":newSessionId})

class UpdateModelForDatasetId(BaseHandler):
    def get(self):
        '''Train a new model (or update) for given dataset ID
        '''
        dsid = self.get_int_arg("dsid",default=0)

        # create feature vectors from database
        f=[];
        for a in self.db.labeledinstances.find({"dsid":dsid}): 
            f.append([float(val) for val in a['feature']])

        # create label vector from database
        l=[];
        for a in self.db.labeledinstances.find({"dsid":dsid}): 
            l.append(a['label'])

        # fit the model to the data
        c1 = KNeighborsClassifier(n_neighbors=1);
        acc = -1;
        if l:
            c1.fit(f,l) # training
            lstar = c1.predict(f)
            self.clf = c1
            acc = sum(lstar==l)/float(len(l))
            bytes = pickle.dumps(c1)
            self.db.models.update({"dsid":dsid},
                {  "$set": {"model":Binary(bytes)}  },
                upsert=True)

        # send back the resubstitution accuracy
        # if training takes a while, we are blocking tornado!! No!!
        self.write_json({"resubAccuracy":acc})

class PredictOneFromDatasetId(BaseHandler):
    def post(self):
        '''Predict the class of a sent feature vector
        '''
        data = json.loads(self.request.body.decode("utf-8"))    

        vals = data['feature'];
        fvals = [float(val) for val in vals];
        fvals = np.array(fvals).reshape(1, -1)
        dsid  = data['dsid']

        # load the model from the database (using pickle)
        # we are blocking tornado!! no!!
        if(self.clf == []):
            print('Loading Model From DB')
            tmp = self.db.models.find_one({"dsid":dsid})
            self.clf = pickle.loads(tmp['model'])
        predLabel = self.clf.predict(fvals);
        self.write_json({"prediction":str(predLabel)})
