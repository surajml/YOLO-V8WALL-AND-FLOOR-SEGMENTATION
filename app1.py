import cv2
import torch
from flask import Flask, request, jsonify, make_response
import numpy as np
from urllib.request import urlopen
from ultralytics import YOLO
import base64

app = Flask(__name__)

model_path = "last42.pt"

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route("/", methods=['GET', 'POST', 'OPTIONS'])
def handle_requests():
    if request.method == 'OPTIONS':
        return handle_preflight()

    if request.method == 'GET':
        return jsonify(message='Welcome to the image processing API!')

    if 'image' not in request.files:
        return jsonify(error='Missing required image file'), 400

    image_file = request.files['image']
    texture_url = request.form.get('texture_url')
    segmentation_option = request.form.get('segmentation_option')

    if not texture_url or segmentation_option not in ['floor', 'wall']:
        return jsonify(error='Invalid request data'), 400

    nparr = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    H, W, _ = img.shape
    model = YOLO(model_path)
    results = model(img)

    result = results[0]
    class_labels = result.boxes.cls.numpy()
    masks = results[0].masks.data
    resized_masks = torch.nn.functional.interpolate(masks.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
    resized_masks = resized_masks.bool()

    response = urlopen(texture_url)
    texture_data = response.read()
    nparr = np.frombuffer(texture_data, np.uint8)
    texture = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    num_repeats_h = int(np.ceil(H / texture.shape[0]))
    num_repeats_w = int(np.ceil(W / texture.shape[1]))
    tiled_texture = np.tile(texture, (num_repeats_h, num_repeats_w, 1))
    tiled_texture = tiled_texture[:H, :W]

    # Perspective transformation for wall texture
    src_points = np.float32([[0, 0], [texture.shape[1], 0], [texture.shape[1], texture.shape[0]], [0, texture.shape[0]]])
    dst_points = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    result_texture = cv2.warpPerspective(tiled_texture, perspective_matrix, (W, H), flags=cv2.INTER_LINEAR)

    # Apply the transformed wall texture to the image
    for mask, label in zip(resized_masks, class_labels):
        if (segmentation_option == 'floor' and label == 1) or (segmentation_option == 'wall' and label == 2):
            mask = mask.numpy()
            img[mask] = result_texture[mask]

    _, buffer = cv2.imencode('.jpg', img)
    encoded_segmented_image = base64.b64encode(buffer).decode('utf-8')

    response = {
        'segmented_image': encoded_segmented_image
    }

    return jsonify(response)

def handle_preflight():
    response = make_response()
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST')
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0')
