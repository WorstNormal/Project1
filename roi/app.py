from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from roi_matching import ImageClassifier
import os

app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')

app.config['UPLOAD'] = upload_folder


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global classifier
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img = os.path.join(app.config['UPLOAD'], filename)
        classifier.add_img(img)
        return render_template('image_render.html', img=img)
    return render_template('image_render.html')


if __name__ == '__main__':
    classifier = ImageClassifier()
    classifier.new_class()
    app.run(debug=True, port=8001)