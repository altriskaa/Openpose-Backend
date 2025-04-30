from flask import Blueprint, request, jsonify
import cv2
import numpy as np
from services.openpose_processor import process_image_with_openpose
from utils.angle_calculator import extract_angles

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict/image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    npimg = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    keypoints, hand_keypoints = process_image_with_openpose(img)
    angles = extract_angles(keypoints, hand_keypoints)

    return jsonify(angles)
