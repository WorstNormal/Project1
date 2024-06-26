from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from roi_matching import ImageClassifier
import os
import cv2
from zipfile import ZipFile

app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')

app.config['UPLOAD'] = upload_folder


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img = os.path.join(app.config['UPLOAD'], filename)

        test_images = 'roi_test'
        classifier = ImageClassifier()
        test_paths = os.listdir(test_images)
        image_class = dict()
        class_image = dict()
        for f in test_paths:
            image = f.split(".")[0]
            image_class[image] = [image]
            class_image[image] = image
            classifier.add_img(os.path.join(test_images, f), f)
        frame = cv2.imread(img)
        result, dist = classifier.predict(frame)
        return render_template('image_render.html', img=img, result=result, dist=dist)
    return render_template('image_render.html')


if __name__ == '__main__':
    app.run(debug=True, port=8001)