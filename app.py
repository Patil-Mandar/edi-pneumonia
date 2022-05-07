from flask import Flask,render_template,request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

app = Flask(__name__)

def preprocess(x):
    img = image.load_img(x, target_size=(32, 32, 3))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img /= 255
    return img

model = tf.keras.models.load_model('basic_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis',methods=['GET','POST'])
def analysis():
    if request.method == 'GET':
        return render_template('analysisform.html')
    
    if request.method == 'POST':
        # img = request.files.get('img','')
        img = request.files['img']
        path = "./static/" + img.filename
        img.save(path)
        img = preprocess(path)
        ans_list = ["Normal", "Pneumonia"]
        ans = np.argmax(model.predict(tf.expand_dims(img, axis=0)))
        return render_template('analysisreport.html',ans=ans_list[ans])


@app.route('/aboutus')
def aboutus():
    return 'About Us'


if __name__ =='__main__':
	app.run(debug = True)