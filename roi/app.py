import zipfile
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from roi_matching import ImageClassifier
import os
import cv2

app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')

app.config['UPLOAD'] = upload_folder


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        file_input = os.path.join(app.config['UPLOAD'], filename)

        data_images = 'roi_data'
        classifier = ImageClassifier()
        test_paths = os.listdir(data_images)
        image_class = dict()
        class_image = dict()
        for f in test_paths:
            image = f.split(".")[0]
            image_class[image] = [image]
            class_image[image] = image
            classifier.add_img(os.path.join(data_images, f), f)
        if file_input.split(".")[1] == "png":
            frame = cv2.imread(file_input)
            result, dist = classifier.predict(frame)
            return render_template('image_render.html', flag=1, flag_1=0,
                                   img=file_input, result=result, dist=dist)
        else:
            list_file = list()
            with zipfile.ZipFile(file_input, 'r') as zip_file:
                zip_file.extractall('static/Image')
            for f in os.listdir('static/Image'):
                frame = cv2.imread(f'static/Image/{f}')
                result, dist = classifier.predict(frame)
                list_file.append({"img": f'static/Image/{f}', 'res':result, "dist":dist})
            print(list_file)
            return render_template('image_render.html', flag=0, flag_1=1,
                                   list_file=list_file)
    return render_template('image_render.html')


if __name__ == '__main__':
    app.run(debug=True, port=8001)