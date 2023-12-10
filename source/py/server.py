
import cv2 
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import thư viện CORS
import base64
from algorithm import *

app = Flask(__name__)
CORS(app, resources={r"/process_image": {"origins": "http://127.0.0.1:5500"}})

@app.route('/process_image', methods=['POST'])
def handle_image():
    try:
        image_data = request.json['image_data']
        type_algorithm = request.json['type_algorithm']
        print(type_algorithm)
        processed_image = None
        if type_algorithm == "gradient_custom":
            processed_image = gradient_custom(image_data)
        
        if type_algorithm == "gradient_library":
            processed_image = gradient_library(image_data)
            
        if type_algorithm == "laplace_custom":
            processed_image = laplace_custom(image_data)
        
        if type_algorithm == "laplace_library":
            processed_image = laplace_library(image_data)

        if type_algorithm == "laplacian_library":
            processed_image = laplacian_library(image_data)
            
        if type_algorithm == "laplacian_custom":
            processed_image = laplacian_custom(image_data)

        if type_algorithm == "canny_library":
            processed_image = canny_library(image_data)
        
        if type_algorithm == "canny_custom":
            processed_image = canny_custom(image_data)
        
        
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