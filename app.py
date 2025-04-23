from flask import Flask, request, jsonify
import os
import base64
from PIL import Image, ImageDraw, ImageFont  # Add ImageDraw here
from io import BytesIO
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO
import json
import time
from flask_cors import CORS
from google.genai import types

# Configure Gemini API (replace with your API key)
from google import genai

app = Flask(__name__)
CORS(app)

client = genai.Client(api_key="AIzaSyCAbdwpZqjJLSMqTWRvmO5Uno8aDo_wLYA")
yolov8_path = "yolov8/runs/detect/train5/weights/best.pt"
fastercnn_path = "fasterRCNNmodal/fasterrcnn_resnet50_epoch_47.pth"
UPLOAD_FOLDER_FASTER_RCNN = "uploads_faster_rcnn"
UPLOAD_FOLDER_YOLOV8 = "uploads_yolov8"
PREDICTION_FOLDER = "predictions"


os.makedirs(UPLOAD_FOLDER_FASTER_RCNN, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_YOLOV8, exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

try:
    font = ImageFont.truetype("arial.ttf", size=20)
except IOError:
    font = ImageFont.load_default()


def load_faster_rcnn_model():
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 8 + 1
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.load_state_dict(torch.load(fastercnn_path, map_location=device))
    model.to(device)
    model.eval()
    return model


faster_rcnn_model = load_faster_rcnn_model()
yolov8_model = YOLO(yolov8_path)


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
        ]
    )
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


def get_gemini_suggestions(pred):
    if pred["class_name"].lower() == "healthy":
        prompt = f"""
        Provide general care tips for healthy cotton plants as an HTML unordered list (<ul> and <li> tags).
        Include advice on watering, fertilization, sunlight, and common maintenance practices.
        Aim for a response within 200 words.
        Confidence: {pred['confidence']:.2f}
        """
    else:
        prompt = f"""
        Provide treatment and prevention methods for **{pred["class_name"]}** in cotton plant.
        Include both organic and chemical solutions.
        Format as bullet points. It should be in html embedded format.
        Format the output as an HTML unordered list (<ul> and <li> tags).
        Format the output within 300-400 words.
        Confidence: {pred['confidence']:.2f}
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )

        return {
            "suggestions": response.text,
        }

    except Exception as e:
        return {"error": f"Couldn't generate the response: {str(e)}"}


def save_predictions(predictions, annotated_image, model_name, file_id):
    # Save prediction data as JSON
    prediction_data_path = os.path.join(
        PREDICTION_FOLDER, f"{model_name}_predictions_{file_id}.json"
    )
    with open(prediction_data_path, "w") as f:
        json.dump(predictions, f, indent=4)

    # Save annotated image
    annotated_image_path = os.path.join(
        PREDICTION_FOLDER, f"{model_name}_annotated_{file_id}.jpg"
    )
    annotated_image.save(annotated_image_path, format="JPEG")


@app.route("/test/fasterrcnn", methods=["POST"])
def test_fasterrcnn():
    if "image" not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file:
        filepath = os.path.join(UPLOAD_FOLDER_FASTER_RCNN, file.filename)
        file.save(filepath)
        try:
            image_tensor = preprocess_image(filepath)
            with torch.no_grad():
                predictions = faster_rcnn_model(image_tensor)[0]
            boxes = predictions["boxes"].cpu().numpy().tolist()
            labels = predictions["labels"].cpu().numpy().tolist()
            scores = predictions["scores"].cpu().numpy().tolist()
            confidence_threshold = 0.5
            filtered_predictions = [
                {"class": label, "confidence": score, "bbox": box}
                for box, label, score in zip(boxes, labels, scores)
                if score >= confidence_threshold
            ]
            from PIL import ImageDraw

            image = Image.open(filepath).convert("RGB")
            draw = ImageDraw.Draw(image)
            for pred in filtered_predictions:
                bbox = pred["bbox"]
                label_id = pred["class"]
                confidence = pred["confidence"]
                disease_name = {
                    0: "crops",
                    1: "PowderyMildew",
                    2: "aphids",
                    3: "army-worm",
                    4: "bacterialblight",
                    5: "curlvirus",
                    6: "healthy",
                    7: "targetspot",
                }[label_id]
                label_text = f"(Faster R-CNN)({disease_name})({confidence:.2f})"
                pred["class_name"] = disease_name
                pred["suggestions"] = get_gemini_suggestions(pred)["suggestions"]
                draw.rectangle(bbox, outline="white", width=2)
                draw.text((bbox[0], bbox[1]), label_text, fill="white", font=font)
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            file_id = int(time.time())
            save_predictions(filtered_predictions, image, "fasterrcnn", file_id)
            # get_gemini_suggestions(filtered_predictions)
            os.remove(filepath)
            return (
                jsonify(
                    {
                        "predictions": filtered_predictions,
                        "image": img_str,
                    }
                ),
                200,
            )
        except Exception as e:
            os.remove(filepath)
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/test/yolov8", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    # Save the uploaded file
    if file:
        filepath = os.path.join(UPLOAD_FOLDER_YOLOV8, file.filename)
        file.save(filepath)
        try:
            # Perform prediction on the uploaded image
            results = yolov8_model.predict(
                source=filepath,
                conf=0.5,  # Confidence threshold
                iou=0.45,  # IoU threshold
                save=False,  # Do not save results to disk
                show=False,  # Do not display results in a window
            )
            # Extract predictions from the results
            predictions = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = yolov8_model.names[class_id]
                    pred = {
                        "class": class_id,  # Class index
                        "class_name": class_name,
                        "confidence": float(box.conf),  # Confidence score
                        "bbox": box.xyxy.tolist()[
                            0
                        ],  # Bounding box coordinates [x1, y1, x2, y2]
                    }
                    pred["suggestions"] = get_gemini_suggestions(pred)["suggestions"]
                    predictions.append(pred)
            # Get the plotted image (annotated with detections)
            annotated_image = results[0].plot()  # Returns an image with bounding boxes
            # Convert the annotated image to Base64
            buffered = BytesIO()
            Image.fromarray(annotated_image).save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            # Save predictions locally
            file_id = int(time.time())  # Generate a unique ID based on timestamp
            save_predictions(
                predictions, Image.fromarray(annotated_image), "yolov8", file_id
            )

            os.remove(filepath)

            return (
                jsonify(
                    {
                        "predictions": predictions,
                        "image": img_str,  # Base64-encoded image
                    }
                ),
                200,
            )
        except Exception as e:
            # Handle any errors during prediction
            os.remove(filepath)  # Clean up the uploaded file even if an error occurs
            print(e)
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/test/comparison", methods=["POST"])
def compare_models():
    # Check if an image file is included in the request
    if "image" not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    file = request.files["image"]
    # Check if a file was selected
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    # Save the uploaded file
    if file:
        filepath = os.path.join(UPLOAD_FOLDER_FASTER_RCNN, file.filename)
        file.save(filepath)
        try:
            # Perform Faster R-CNN prediction
            image_tensor = preprocess_image(filepath)
            with torch.no_grad():  # Disable gradient computation
                faster_rcnn_predictions = faster_rcnn_model(image_tensor)[
                    0
                ]  # Get predictions

            # Extract Faster R-CNN predictions
            faster_rcnn_boxes = faster_rcnn_predictions["boxes"].cpu().numpy().tolist()
            faster_rcnn_labels = (
                faster_rcnn_predictions["labels"].cpu().numpy().tolist()
            )
            faster_rcnn_scores = (
                faster_rcnn_predictions["scores"].cpu().numpy().tolist()
            )

            # Filter Faster R-CNN predictions based on a confidence threshold
            confidence_threshold = 0.5
            faster_rcnn_filtered = [
                {
                    "model": "fasterrcnn",
                    "class": label,
                    "confidence": score,
                    "bbox": box,
                }
                for box, label, score in zip(
                    faster_rcnn_boxes, faster_rcnn_labels, faster_rcnn_scores
                )
                if score >= confidence_threshold
            ]

            # Perform YOLOv8 prediction
            results = yolov8_model.predict(
                source=filepath,
                conf=0.5,  # Confidence threshold
                iou=0.45,  # IoU threshold
                save=False,  # Do not save results to disk
                show=False,  # Do not display results in a window
            )

            # Extract YOLOv8 predictions
            yolov8_filtered = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = yolov8_model.names[class_id]
                    pred = {
                        "model": "yolov8",
                        "class": class_id,  # Class index
                        "confidence": float(box.conf),  # Confidence score
                        "class_name": class_name,
                        "bbox": box.xyxy.tolist()[
                            0
                        ],  # Bounding box coordinates [x1, y1, x2, y2]
                    }
                    yolov8_filtered.append(pred)

            # Combine predictions from both models
            all_predictions = faster_rcnn_filtered + yolov8_filtered

            # Group predictions by bounding box similarity
            def bbox_iou(box1, box2):
                """Calculate Intersection over Union (IoU) between two bounding boxes."""
                x1_min, y1_min, x1_max, y1_max = box1
                x2_min, y2_min, x2_max, y2_max = box2
                inter_xmin = max(x1_min, x2_min)
                inter_ymin = max(y1_min, y2_min)
                inter_xmax = min(x1_max, x2_max)
                inter_ymax = min(y1_max, y2_max)
                inter_area = max(0, inter_xmax - inter_xmin) * max(
                    0, inter_ymax - inter_ymin
                )
                box1_area = (x1_max - x1_min) * (y1_max - y1_min)
                box2_area = (x2_max - x2_min) * (y2_max - y2_min)
                union_area = box1_area + box2_area - inter_area
                return inter_area / union_area if union_area > 0 else 0

            best_predictions = []
            used_indices = set()

            for i, pred1 in enumerate(all_predictions):
                if i in used_indices:
                    continue
                best_pred = pred1
                for j, pred2 in enumerate(all_predictions):
                    if j in used_indices or j == i:
                        continue
                    # Compare bounding boxes using IoU
                    iou = bbox_iou(pred1["bbox"], pred2["bbox"])
                    if iou > 0.5:  # If bounding boxes overlap significantly
                        used_indices.add(j)
                        # Choose the prediction with the higher confidence score
                        if pred2["confidence"] > best_pred["confidence"]:
                            best_pred = pred2
                used_indices.add(i)
                best_predictions.append(best_pred)

            # Annotate the image with the best predictions
            image = Image.open(filepath).convert("RGB")
            draw = ImageDraw.Draw(image)
            for pred in best_predictions:
                bbox = pred["bbox"]
                confidence = pred["confidence"]

                # Get disease name based on model type
                if pred["model"] == "fasterrcnn":
                    disease_name = {
                        0: "crops",
                        1: "PowderyMildew",
                        2: "aphids",
                        3: "army-worm",
                        4: "bacterialblight",
                        5: "curlvirus",
                        6: "healthy",
                        7: "targetspot",
                    }[pred["class"]]
                    pred["class_name"] = disease_name
                else:  # YOLOv8
                    disease_name = pred["class_name"]

                # Create label and draw
                label_text = f"({pred['model']})({disease_name})({confidence:.2f})"
                draw.rectangle(bbox, outline="white", width=2)
                draw.text((bbox[0], bbox[1]), label_text, fill="white", font=font)

                pred["suggestions"] = get_gemini_suggestions(pred)["suggestions"]

            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            os.remove(filepath)

            return (
                jsonify(
                    {
                        "best_predictions": best_predictions,
                        "image": img_str,
                    }
                ),
                200,
            )

        except Exception as e:
            os.remove(filepath)
            return jsonify({"error": f"Comparison failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
