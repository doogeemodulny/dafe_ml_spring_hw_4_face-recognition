from flask import Flask, request, jsonify
import numpy as np
import torch
import os
from PIL import Image

from train import FaceNetModel
from evaluation import evaluate_similarity


model = FaceNetModel()
model.load_state_dict(torch.load("best_model.pth"))
model.eval()


app = Flask(__name__)


@app.route('/', methods=['GET'])
def lending():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Face Recognition API</title>
        <style>
            * {
                box-sizing: border-box;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
            }

            body {
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }

            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }

            h1 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                margin-top: 0;
            }

            h2 {
                color: #34495e;
                margin-top: 30px;
            }

            code {
                background: #f8f9fa;
                padding: 2px 5px;
                border-radius: 3px;
                font-family: monospace;
            }

            pre {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
            }

            a.example-link {
                display: inline-block;
                background: #3498db;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                text-decoration: none;
                margin: 10px 0;
                transition: background 0.3s;
            }

            a.example-link:hover {
                background: #2980b9;
            }

            .note {
                background: #fff3cd;
                padding: 15px;
                border-left: 4px solid #ffc107;
                margin: 20px 0;
            }

            .response-sample {
                background: #e9f5ff;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
            }

            .error-codes {
                background: #f8d7da;
                padding: 15px;
                border-left: 4px solid #dc3545;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎭 Face Similarity Recognition Service</h1>

            <div class="note">
                <strong>📌 Note:</strong> Ensure absolute paths or paths relative to server working directory
            </div>

            <h2>📖 API Usage Instructions</h2>
            <p>Send GET request to <code>/predict</code> endpoint with two image paths:</p>

            <pre>http://localhost:5000/predict?image1=PATH_TO_IMAGE1&image2=PATH_TO_IMAGE2</pre>

            <h3>🔍 Example Request:</h3>
            <a class="example-link" href="/predict?image1=test_images/edik/1.jpg&image2=test_images/edik/2.jpg">
                Try Example Request
            </a>

            <h3>📄 Response Format:</h3>
            <div class="response-sample">
                <pre>{
    "image1": "input_image_path",
    "image2": "comparison_image_path", 
    "similarity": 0.757
}</pre>
            </div>

            <h3>🚨 Error Codes:</h3>
            <div class="error-codes">
                <ul>
                    <li><strong>400</strong> - Missing image paths</li>
                    <li><strong>404</strong> - Image file not found</li>
                    <li><strong>500</strong> - Internal server error</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """


from io import BytesIO
import base64


@app.route('/predict', methods=['GET'])
def predict():
    try:
        im1_path = request.args.get('image1')
        im2_path = request.args.get('image2')

        # Проверка наличия путей
        if not im1_path or not im2_path:
            return jsonify({'error': 'Missing image paths'}), 400

        # Проверка существования файлов
        if not os.path.exists(im1_path):
            return jsonify({'error': f'File not found: {im1_path}'}), 404
        if not os.path.exists(im2_path):
            return jsonify({'error': f'File not found: {im2_path}'}), 404

        # Конвертация изображений в base64
        def encode_image(image_path):
            try:
                with Image.open(image_path) as img:
                    img = img.convert('RGB')  # Конвертация в RGB
                    buffer = BytesIO()
                    img.save(buffer, format="JPEG")
                    return base64.b64encode(buffer.getvalue()).decode('utf-8')
            except Exception as e:
                raise ValueError(f"Error processing {image_path}: {str(e)}")

        im1_b64 = encode_image(im1_path)
        im2_b64 = encode_image(im2_path)

        # Вычисление схожести
        similarity = evaluate_similarity(model, im1_path, im2_path)

        # Определяем формат ответа
        response_format = request.args.get('format', 'html')

        if response_format == 'json':
            return jsonify({
                'image1': im1_path,
                'image2': im2_path,
                'similarity': float(similarity)
            })

        # HTML-ответ с изображениями
        return f"""
        <html>
            <head>
                <title>Face Recognition Result</title>
                <style>
                    .container {{ 
                        display: flex; 
                        gap: 20px; 
                        justify-content: center;
                        margin: 20px;
                    }}
                    .image-box {{ 
                        text-align: center; 
                        padding: 10px;
                        border: 1px solid #ddd;
                        border-radius: 8px;
                    }}
                    img {{ 
                        max-width: 400px; 
                        max-height: 400px;
                        object-fit: contain;
                    }}
                    .result {{ 
                        text-align: center; 
                        font-size: 24px;
                        margin: 20px;
                    }}
                </style>
            </head>
            <body>
                <h1 style="text-align: center;">Face Similarity Result</h1>
                <div class="container">
                    <div class="image-box">
                        <img src="data:image/jpeg;base64,{im1_b64}">
                        <p>{im1_path}</p>
                    </div>
                    <div class="image-box">
                        <img src="data:image/jpeg;base64,{im2_b64}">
                        <p>{im2_path}</p>
                    </div>
                </div>
                <div class="result">
                    Similarity Score: <strong>{similarity:.3f}</strong>
                </div>
                <div style="text-align: center; margin-top: 20px;">
                    <a href="/" style="padding: 10px; background: #007bff; color: white; text-decoration: none; border-radius: 5px;">
                        Back to Main Page
                    </a>
                </div>
            </body>
        </html>
        """

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)