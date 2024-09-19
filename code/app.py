import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import cv2

app = Flask(__name__)
CORS(app)

# Load the YOLOv8 model once when the server starts
model = YOLO('best.pt')  # You can replace 'yolov8n.pt' with your own trained YOLOv8 model

# Ensure the 'uploads' and 'outputs' directories exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

if not os.path.exists('outputs'):
    os.makedirs('outputs')

@app.route('/')
def home():
    """
    Default route for the root URL.
    :return: A simple webpage to upload an image.
    """
    return '''
    <!doctype html>
    <html lang="en">
    <head>
        <title>YOLOv8 Image Inference</title>
    </head>
    <body>
        <h1>Upload an Image for YOLOv8 Inference</h1>
        <form action="/image" method="POST" enctype="multipart/form-data">
            <label for="file">Choose an image:</label>
            <input type="file" id="file" name="file" accept="image/*">
            <br><br>
            <input type="submit" value="Upload">
        </form>
    </body>
    </html>
    '''

@app.route('/status', methods=['GET'])
def status():
    """
    This is a health check endpoint to check if the server is running.
    :return: A JSON object with a key "result" and value "ok".
    """
    return jsonify({"result": "ok"})

@app.route('/image', methods=['POST'])
def image_predict():
    """
    This is the main endpoint for image prediction using YOLOv8.
    :return: HTML page showing the predictions with the image and bounding boxes.
    """
    try:
        # Handle file upload
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join('uploads', filename)
        
        # Save the uploaded image to the 'uploads' directory
        file.save(file_path)
        
        # Run inference on the image using the YOLOv8 model
        results = model(file_path)
        
        # Define the label to ID mapping
        label_to_id = {
            "NA": 'NA',
            "Bullseye": 10,
            "one": 11,
            "two": 12,
            "three": 13,
            "four": 14,
            "five": 15,
            "six": 16,
            "seven": 17,
            "eight": 18,
            "nine": 19,
            "AlphabetA": 20,
            "AlphabetB": 21,
            "AlphabetC": 22,
            "AlphabetD": 23,
            "AlphabetE": 24,
            "AlphabetF": 25,
            "AlphabetG": 26,
            "AlphabetH": 27,
            "AlphabetS": 28,
            "AlphabetT": 29,
            "AlphabetU": 30,
            "AlphabetV": 31,
            "AlphabetW": 32,
            "AlphabetX": 33,
            "AlphabetY": 34,
            "AlphabetZ": 35,
            "up Arrow": 36,
            "down Arrow": 37,
            "right Arrow": 38,
            "left Arrow": 39,
            "Stop": 40
        }

        # Load the image using OpenCV
        img = cv2.imread(file_path)

        predictions = []
        for result in results:
            boxes = result.boxes  # Boxes object holds the bounding boxes, labels, and scores
            for box in boxes:
                # Each box has xyxy, confidence, and class (label)
                xyxy = box.xyxy[0].tolist()  # Coordinates of the bounding box
                conf = box.conf[0].tolist()  # Confidence score
                cls = box.cls[0].tolist()    # Class label

                label_name = result.names[int(cls)]  # Get label name from YOLO result
                label_id = label_to_id.get(label_name, "NA")  # Map label name to ID using the dictionary

                predictions.append({
                    'label': label_name,
                    'label_id': label_id,  # Insert the mapped label ID
                    'confidence': float(conf),
                    'bbox': [int(x) for x in xyxy]
                })

                # Draw bounding box and label on the image
                x1, y1, x2, y2 = [int(c) for c in xyxy]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{label_name} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Save the image with bounding boxes in 'outputs' directory
        output_image_path = os.path.join('outputs', f'pred_{filename}')
        cv2.imwrite(output_image_path, img)

        # Generate HTML to display the image and predictions
        prediction_html = f"<h2>Predictions:</h2><ul>"
        for pred in predictions:
            prediction_html += f"<li>Label: {pred['label']}, Confidence: {pred['confidence']:.2f}, BBox: {pred['bbox']}</li>"
        prediction_html += "</ul>"

        return f'''
        <!doctype html>
        <html lang="en">
        <head>
            <title>YOLOv8 Image Inference</title>
        </head>
        <body>
            <h1>YOLOv8 Image Inference Results</h1>
            {prediction_html}
            <h2>Processed Image:</h2>
            <img src="/output_image/{filename}" alt="Processed Image">
            <br><br>
            <a href="/">Upload another image</a>
        </body>
        </html>
        '''

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)})

@app.route('/output_image/<filename>')
def output_image(filename):
    """
    Serve the image with bounding boxes.
    """
    return send_from_directory('outputs', f'pred_{filename}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
