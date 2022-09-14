import os
import time
from pathlib import PurePath
from flask import Flask, flash, request, redirect, url_for
from flask import send_file, send_from_directory
from werkzeug.utils import secure_filename

import compressai
from examples.codec import _encode_checkpoint, _decode_checkpoint



UPLOAD_FOLDER = '/home/zhangyiwei/hdd/workspace/VCIP_Challenge/TinyLIC/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs("uploads/inputs", exist_ok=True)
os.makedirs("uploads/outputs", exist_ok=True)
CKPT_PATH = 'pretrained/elic/best/lambda0.015.pth.tar'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000


# def text_hash(txt, length=6):
#     return hashlib.md5(txt).hexdigest()[:length]


def time_hash():
    return hex(int(time.time()))[2:]


def allowed_image_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_bin_file(filename):
    return filename.endswith(".bin")


@app.route('/limbocodec', methods=['GET', 'POST'])
def limbocodec():
    return '''
    <!doctype html>
    <title>Limbo Codec</title>
    <h1>Limbo Codec</h1>
    <h2>Encode Image</h2>
    <p> 'png', 'jpg', 'jpeg' format is allowed. </p>
    <p> image size exceed 4K need to be processed in chunks, which may take more time .</p>
    <form action=/limbocodec/encode method=post enctype=multipart/form-data>
      <input type=file accept="image/*" name=file>
      <input type=submit value=Encode>
    </form>
    <h2>Decode Bitstream</h2>
    <p> 'bin' format is allowed. </p>
    <form action=/limbocodec/decode method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Decode>
    </form>
    '''


@app.route('/limbocodec/uploads/<name>')
def download_file(name):
    return send_file(os.path.join(app.config["UPLOAD_FOLDER"], 'outputs', name))


app.add_url_rule(
    "/uploads/<name>", endpoint="download_file", build_only=True
)


@app.post("/limbocodec/encode")
def request_encode():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_image_file(file.filename):
        _path = PurePath(secure_filename(file.filename))
        stamp = time_hash()
        in_name = f"{_path.stem}_{stamp}{_path.suffix}"
        out_name = f"{_path.stem}_{stamp}.bin"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], "inputs", in_name)
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], "outputs", out_name)
        file.save(in_path)
        try:
            encode_func(in_path, out_path)
        except Exception as e:
            return "Encode Error:" + e.__repr__
        else:
            return send_file(out_path)
    else:
        return "Decode Error: Not allowed file."


@app.post("/limbocodec/decode")
def request_decode():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_bin_file(file.filename):
        _path = PurePath(secure_filename(file.filename))
        stamp = time_hash()
        in_name = f"{_path.stem}_{stamp}{_path.suffix}"
        out_name = f"{_path.stem}_{stamp}.png"
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], "inputs", in_name)
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], "outputs", out_name)
        file.save(in_path)
        try:
            decode_func(in_path, out_path)
        except Exception as e:
            return "Decode Error: " + e.__repr__
        else:
            return redirect(url_for('download_file', name=out_name))
            # return send_file(out_path)
    else:
        return "Decode Error: Not allowed file."


def encode_func(fin, fout):
    encoder_kwargs = {
    "input":fin,
    "model":'elic',
    "num_of_frames":1,
    "metric":'mse',
    "quality": 3,
    "ckpt_path":CKPT_PATH,
    "coder": compressai.available_entropy_coders()[0],
    "device":'cuda',
    "output":fout
    }
    _encode_checkpoint(**encoder_kwargs)
    return


def decode_func(fin, fout):
    decoder_kwargs = {
        "inputpath":fin,
        "coder": compressai.available_entropy_coders()[0],
        "ckpt_path":CKPT_PATH,
        "show" : False,
        "device":"cuda",
        "output":fout
    }
    _decode_checkpoint(**decoder_kwargs)