from flask import Flask, request
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

# Load the pre-trained H4 model
model = tf.keras.models.load_model('/Users/harshikahooda/Downloads/final_vgg1930epochs.h5',compile=False)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file from the form data
    image_file = request.files['image']
    
    # Read the image file and preprocess it
    img = cv2.resize(cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1), (32, 32)) / 255.0
    
    # Make predictions using the H4 model
    prediction = model.predict(img.reshape(1, 32, 32, 3))
    #print(prediction.argmax())
    desired_class = prediction.argmax()
    probability = prediction[0][desired_class]
    
    
    # Return the predicted results
    result=prediction.argmax()
    print(result)
    print("probability is :",probability)
    classes = {
        0:'Acne/Rosacea',
        1:'Actinic Keratosis/Basal Cell Carcinoma/Malignant Lesions',
        2:'Eczema',
        3:'Melanoma Skin Cancer/Nevi/Moles',
        4:'Psoriasis/Lichen Planus and related diseases', 
        5:'Tinea Ringworm/Candidiasis/Fungal Infections',
        6:'Urticaria/Hives', 
        7:'Nail Fungus/Nail Disease'
        }
    return classes[result]
    # return str(prediction)

if __name__ == '__main__':
    app.run(debug=True)
