import os
from keras.models import load_model
from flask import json, Flask, flash, request, redirect, render_template, jsonify
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import tensorflow_hub as hub
from gevent.pywsgi import WSGIServer

application = Flask(__name__)
UPLOAD_FOLDER = './uploads/'
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_ex(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in EXTENSIONS

def req(y_true, y_pred):
    req = tf.metrics.req(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return req

# model1 = load_model("model_skin_diseases.h5")
# model2 = load_model("model_skin_types.h5")

model1 = tf.keras.models.load_model(
       ("model_skin_diseases.h5"),
       custom_objects={'KerasLayer':hub.KerasLayer}
)
model2 = tf.keras.models.load_model(
       ("model_skin_types.h5"),
       custom_objects={'KerasLayer':hub.KerasLayer}
)

@application.route('/recomendation', methods = ['GET','POST'])
def predict1():
    # print(request.files)
    # data = json.loads(request.data)
    print(request.files)
    print(request.form)
    # print(request.values)
    # print(request.get_json(force=True))

    # if 'file' not in request.files:
    #     respond = jsonify({'message': 'No image'})
    #     respond.status_code = 400
    #     return respond
    files = {'filename':'download.jpg'}
    if 'file' in request.files:
        files = request.files.getlist('file')
    errors = {}
    # success = False
    # for file in files:
    #     if file and allowed_ex(file.filename):
    #         file.save(os.path.join(application.config['UPLOAD_FOLDER'], file.filename))
    #         success = True
    #     else:
    #         errors["message"] = 'The type of {} is wrong'.format(file.filename)
    # if not success:
    #     resp = jsonify(errors)
    #     resp.status_code = 400
    #     return resp
    
    
    foto = os.path.join(application.config['UPLOAD_FOLDER'], 'download.jpeg')
    img = keras.utils.load_img(foto, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model1.predict(images, batch_size=10)
    classes2 = model2.predict(images, batch_size=10)
    value2 = max(classes2[0])
    value = max(classes[0])
    i = 0
    index_value=0
    for result in classes[0]:
        if result == value:
            index_value = i
        i += 1
    skin_value = ""
    if index_value == 0:
        skin_value = "Acne"
    elif index_value == 1:
        skin_value = "Black Spots"
    elif index_value == 2:
        skin_value = "Puff Eyes"
    elif index_value == 3:
        skin_value = "Wrinkles"
    print(classes2[0])
    index = np.argmax(classes2[0])
    print(index)
    skin_value2 = ""
    if index == 0:
        skin_value2 = "Dry"
    elif index == 1:
        skin_value2 = "Normal"
    elif index == 2:
        skin_value2 = "Oily"
    elif index == 3:
        skin_value2 = "Sensitive"
   
    return jsonify(
         {
                "status": "Success",
                "message": "Successfully making prediction",
                "data": {
                    "input1": skin_value,
                    "input2": skin_value2,
                }
            }
        )

# @application.route('/recomendation', methods = ['GET'])
# def predict2():
#     print(request)
#     if 'file' not in request.files:
#         respond = jsonify({'message': 'No image'})
#         respond.status_code = 400
#         return respond
#     files = request.files.getlist('file')
#     filename = "download.jpg"
#     errors = {}
#     success = False
#     for file in files:
#         if file and allowed_ex(file.filename):
#             file.save(os.path.join(application.config['UPLOAD_FOLDER'], file.filename))
#             success = True
#         else:
#             errors["message"] = 'The type of {} is wrong'.format(file.filename)
#     if not success:
#         resp = jsonify(errors)
#         resp.status_code = 400
#         return resp
    
    
#     foto = os.path.join(application.config['UPLOAD_FOLDER'], file.filename)
#     img = keras.utils.load_img(foto, target_size=(224, 224))
#     x = tf.keras.utils.img_to_array(img)
#     x /= 255
#     x = np.expand_dims(x, axis=0)
#     images = np.vstack([x])
#     classes = model1.predict(images, batch_size=10)
#     classes2 = model2.predict(images, batch_size=10)
#     value2 = max(classes2[0])
#     value = max(classes[0])
#     i = 0
#     index_value=0
#     for result in classes[0]:
#         if result == value:
#             index_value = i
#         i += 1
#     skin_value = ""
#     if index_value == 0:
#         skin_value = "Acne"
#     elif index_value == 1:
#         skin_value = "Black Spots"
#     elif index_value == 2:
#         skin_value = "Puff Eyes"
#     elif index_value == 3:
#         skin_value = "Wrinkles"
#     print(classes2[0])
#     index = np.argmax(classes2[0])
#     print(index)
#     skin_value2 = ""
#     if index == 0:
#         skin_value2 = "Dry"
#     elif index == 1:
#         skin_value2 = "Normal"
#     elif index == 2:
#         skin_value2 = "Oily"
#     elif index == 3:
#         skin_value2 = "Sensitive"
   
#     return jsonify(
#          {
#                 "status": "Success",
#                 "message": "Successfully making prediction",
#                 "data": {
#                     "input1": skin_value,
#                     "input2": skin_value2,
#                 }
#             }
#         )

if __name__ == "__main__":
    application.run(debug=True, host = '127.0.0.1', port = 5000)
    # application.debug = True 
    # http_server = WSGIServer(('', 5000), application)
    # http_server.serve_forever()


    


    
