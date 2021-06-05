import os
from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
from model import Model


UPLOAD_FOLDER = 'D:/PycharmProjects/ML_Project/static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


model = Model()


@app.route('/')
def home():
    return render_template('main.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    name = request.form.get('name')
    if file.filename == '':
        flash('Изображение не выбрано')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        faces = model.add_photo(file, name)
        return render_template('main.html', filename=filename, faces=faces)
    else:
        flash('Разрешены только: png, jpg, jpeg')
        return redirect(request.url)


if __name__ == '__main__':
  app.run(debug=True)
