from flask import Flask,render_template,request
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import cv2

app = Flask(__name__)

def detect(model, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))
    img = np.array(img).reshape(-1, 150, 150, 1)
    img = img / 255.
    img = np.array(img)

    pred = model.predict([ img ])
    return 1 if pred[0][0]>0.5 else 0

model = tf.keras.models.load_model('pneumonia_detector.h5')




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis',methods=['GET','POST'])
def analysis():
    if request.method == 'GET':
        return render_template('analysisform.html')
    
    if request.method == 'POST':
        img = request.files['img']
        path = "./static/" + img.filename
        img.save(path)

        ans_list = ["Normal", "Pneumonia"]
        ans = detect(model, path)

        return render_template('analysisform.html',ans=ans_list[ans], index=ans, path=path)


@app.route('/aboutus')
def aboutus():
    return 'About Us'


if __name__ =='__main__':
	app.run(debug = True)

