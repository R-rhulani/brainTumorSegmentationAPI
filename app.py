import cv2
from flask import Flask, request, jsonify, Response
import numpy as np
from unetClass import SimpleUNetLayer  # Assuming you have defined your SimpleUNetLayer class
from flask_cors import CORS  # Import the CORS module

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})

# Load the trained weights from the numpy array file
loaded_weights = np.load("trained_weights.npz")

# Create an instance of SimpleUNetLayer
model = SimpleUNetLayer()
model.weights = loaded_weights["layer3_weights"]


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the .npy image file from the POST request
        image_file = request.files['image']
        image_data = np.load(image_file)

        # Ensure that the input image has the correct shape (128, 128, 128, 3)
        if image_data.shape != (128, 128, 128, 3):
            return jsonify({'error': 'Invalid image shape. Expected (128, 128, 128, 3).'})

        # Perform forward pass to get predictions
        predictions = model.forward(image_data)

        # Convert predictions to a JPEG image
        prediction_image = np.argmax(predictions, axis=3)[0, :, :]
        prediction_image = (prediction_image * 255).astype(np.uint8)

        # Create a JPEG response
        _, image_encoded = cv2.imencode('.jpg', prediction_image)
        response = Response(response=image_encoded.tobytes(), content_type='image/jpeg')
        return response

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
