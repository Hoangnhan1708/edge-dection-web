import numpy as np
import cv2 
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import thư viện CORS
import base64

app = Flask(__name__)
CORS(app, resources={r"/process_image": {"origins": "http://127.0.0.1:5500"}})

def process_image(image_data):
    try:
        _, image_data = image_data.split(';base64,')
        image_data = base64.b64decode(image_data)
        nparr = np.frombuffer(image_data, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img_np is None:
            return None
        
        laplacian_img = cv2.Laplacian(img_np, cv2.CV_8U, ksize=3)
        
        # Trả về ảnh đã xử lý dưới dạng mảng NumPy
        return laplacian_img
        
    except Exception as e:
        return str(e)

@app.route('/process_image', methods=['POST'])
def handle_image():
    try:
        image_data = request.json['image_data']
        processed_image = process_image(image_data)
        
        if processed_image is None:
            return jsonify({'error': 'Không thể xử lý ảnh'}), 500
        
        # Chuyển ảnh đã xử lý thành dạng base64 để trả về
        retval, buffer = cv2.imencode('.png', processed_image)
        
        if not retval:
            return jsonify({'error': 'Không thể mã hóa ảnh'}), 500
        
        encoded_image = base64.b64encode(buffer).decode()
        
        return jsonify({'processed_image': encoded_image})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

app.config['DEBUG'] = True

if __name__ == '__main__':
    app.run(debug=True)