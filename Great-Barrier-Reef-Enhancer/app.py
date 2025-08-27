# Updated app.py with new route for decision support page
from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import cv2
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from werkzeug.utils import secure_filename
import random
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

CKPT_PATH = 'best.pt'
IMG_SIZE = 640  # Standard YOLOv5 input size
CONF = 0.25
IOU = 0.40
AUGMENT = False

def load_model(ckpt_path, conf=0.25, iou=0.50):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=ckpt_path, force_reload=True, trust_repo=True)
    model.conf = conf
    model.iou = iou
    model.classes = None
    model.multi_label = False
    model.max_det = 1000
    return model

model = load_model(CKPT_PATH, conf=CONF, iou=IOU)

np.random.seed(32)
colors = [(0, 255, 0), (0, 0, 255)]  # Green and red for visibility

def predict(model, img, size=640, augment=False):
    height, width = img.shape[:2]
    print(f"Original image size: {width}x{height} at {time.strftime('%Y-%m-%d %H:%M EDT', time.gmtime(time.time() - 4*3600))}")
    img_resized = cv2.resize(img, (size, size))
    print(f"Resized image size: {size}x{size}")
    results = model(img_resized, size=size, augment=augment)
    preds = results.pandas().xyxy[0]
    print(f"Number of raw detections: {len(preds)}")
    bboxes = preds[['xmin', 'ymin', 'xmax', 'ymax']].values
    if len(bboxes):
        scale_x = width / size
        scale_y = height / size
        bboxes = bboxes * [scale_x, scale_y, scale_x, scale_y]
        bboxes = bboxes.astype(int)
        confs = preds.confidence.values
        print(f"Raw bboxes: {bboxes}")
        print(f"Raw confidences: {confs}")
        return bboxes, confs
    print("No detections found.")
    return [], []

def voc2coco(bboxes, height, width):
    coco_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        coco_bboxes.append([x1, y1, w, h])
    return np.array(coco_bboxes)

def format_prediction(bboxes, confs):
    annot = ''
    if len(bboxes) > 0:
        for idx in range(len(bboxes)):
            xmin, ymin, w, h = bboxes[idx]
            conf = confs[idx]
            annot += f'{conf} {xmin} {ymin} {w} {h} '
        annot = annot.strip(' ')
    return annot

def draw_bboxes(img, bboxes, confs, classes, class_ids, class_name=True, colors=None, bbox_format='coco', line_thickness=2):
    img = img.copy()
    marked_bboxes = []
    if len(bboxes) == 0:
        print("No bounding boxes to draw.")
        return img
    for i, (bbox, conf) in enumerate(zip(bboxes, confs)):
        if bbox_format == 'coco':
            x, y, w, h = map(int, bbox)
            if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
                print(f"Invalid bbox coordinates: {bbox}, skipping.")
                continue
            # Custom marking logic: Draw if confidence > 0.5 with 50% probability
            if conf > 0.5:
                if random.random() < 0.5:  # 50% chance to mark
                    cv2.rectangle(img, (x, y), (x + w, y + h), colors[i % len(colors)], line_thickness)
                    if class_name:
                        cv2.putText(img, classes[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[i % len(colors)], 2)
                    marked_bboxes.append(bbox)
                    print(f"Marked bbox {i} with confidence {conf}: {bbox}")
            else:
                print(f"Skipped bbox {i} with confidence {conf} < 0.5")
    print(f"Drawn {len(marked_bboxes)} out of {len(bboxes)} possible bounding boxes.")
    return img

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = cv2.imread(filepath)[..., ::-1]
            bboxes, confs = predict(model, img, size=IMG_SIZE, augment=AUGMENT)
            annot = format_prediction(bboxes, confs)
            processed_img = draw_bboxes(img, bboxes, confs, ['starfish'] * len(bboxes), list(range(len(bboxes))))

            processed_filename = 'processed_' + filename
            processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
            cv2.imwrite(processed_filepath, processed_img[..., ::-1])

            return render_template('result.html', original=filename, processed=processed_filename, annotation=annot)

    return render_template('index.html')

@app.route('/display/<folder>/<filename>')
def display_image(folder, filename):
    if folder == 'uploads':
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    elif folder == 'processed':
        return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename))
    return 'Image not found', 404

@app.route('/decision_support', methods=['GET'])
def decision_support():
    return render_template('decision_support.html')

if __name__ == '__main__':
    app.run(debug=True)