from flask import Flask, request, jsonify
import os
import base64
from PIL import Image
from io import BytesIO  # Import BytesIO from the io module
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the Faster R-CNN model
def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 11
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.load_state_dict(torch.load("fasterRCNNmodal/fasterrcnn_resnet50_epoch_50.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Image preprocessing function
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

@app.route('/test/fasterrcnn', methods=['POST'])
def test_fasterrcnn():
    # Check if an image file is included in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']

    # Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        try:
            # Preprocess the image
            image_tensor = preprocess_image(filepath)

            # Perform prediction
            with torch.no_grad():  # Disable gradient computation
                predictions = model(image_tensor)[0]  # Get predictions

            # Extract relevant information
            boxes = predictions['boxes'].cpu().numpy().tolist()
            labels = predictions['labels'].cpu().numpy().tolist()
            scores = predictions['scores'].cpu().numpy().tolist()

            # Filter predictions based on a confidence threshold
            confidence_threshold = 0.5
            filtered_predictions = [
                {
                    'class': label,
                    'confidence': score,
                    'bbox': box
                }
                for box, label, score in zip(boxes, labels, scores)
                if score >= confidence_threshold
            ]

            # Plot the annotated image
            from PIL import ImageDraw
            image = Image.open(filepath).convert("RGB")
            draw = ImageDraw.Draw(image)
            for pred in filtered_predictions:
                bbox = pred['bbox']
                draw.rectangle(bbox, outline="red", width=2)
                draw.text((bbox[0], bbox[1]), f"{pred['class']} {pred['confidence']:.2f}", fill="red")

            # Convert the annotated image to Base64
            buffered = BytesIO()  # Use BytesIO to create an in-memory buffer
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Clean up the uploaded file
            os.remove(filepath)

            # Return the predictions and the Base64-encoded image
            return jsonify({
                'predictions': filtered_predictions,
                'image': img_str  # Base64-encoded image
            }), 200

        except Exception as e:
            # Handle any errors during prediction
            os.remove(filepath)  # Clean up the uploaded file even if an error occurs
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
