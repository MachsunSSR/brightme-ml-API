# Import .h5 model
import tensorflow as tf
from keras.models import load_model

model_skin_diseases = load_model('model_skin_diseases.h5')
model_skin_type = load_model('model_skin_type.h5')

