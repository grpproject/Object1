
import uuid

from pathlib import Path
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from PIL import Image

from ai_engine.segmenter import ObjectSegmenter
from ai_engine.feature_extractor import FeatureExtractor
from ai_engine.recognizer import ObjectRecognizer
from ai_engine.database import initialize_database, save_object


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
EXTRACTED_FOLDER = BASE_DIR / "static" / "extracted_objects"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
EXTRACTED_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

initialize_database()

segmenter = ObjectSegmenter()
feature_extractor = FeatureExtractor()
recognizer = ObjectRecognizer(similarity_threshold=0.85)


def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def save_upload(file):

    extension = file.filename.rsplit(".", 1)[1].lower()

    filename = f"{uuid.uuid4().hex}.{extension}"

    filename = secure_filename(filename)

    image_path = UPLOAD_FOLDER / filename

    file.save(image_path)

    try:
        with Image.open(image_path) as image:
            image.verify()

    except Exception:
        image_path.unlink(missing_ok=True)
        raise ValueError("Uploaded file is not a valid image.")

    return image_path


def image_url(image_path):

    relative_path = Path(image_path).relative_to(
        BASE_DIR / "static"
    )

    return "/static/" + relative_path.as_posix()


@app.route("/")
def index():
    return render_template("index.html")


# TRAINING: EXTRACT AND STORE OBJECTS
@app.route("/api/extract", methods=["POST"])
def extract_objects():

    if "image" not in request.files:
        return jsonify({"error": "Image is required."}), 400

    file = request.files["image"]

    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Unsupported image format."}), 400

    try:

        image_path = save_upload(file)

        extracted = segmenter.extract_objects(
            image_path,
            EXTRACTED_FOLDER / "training"
        )

        saved_objects = []

        for obj in extracted:

            embedding = feature_extractor.extract_features(
                obj["image_path"]
            )

            save_object(
                obj["object_id"],
                obj["name"],
                obj["image_path"],
                embedding
            )

            saved_objects.append({
                "object_id": obj["object_id"],
                "name": obj["name"],
                "confidence": obj["confidence"],
                "image_url": image_url(obj["image_path"])
            })

        return jsonify({
            "success": True,
            "message": "Object extraction completed.",
            "total_objects": len(saved_objects),
            "objects": saved_objects
        })

    except Exception as error:

        app.logger.exception("Extraction failed")

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# PREDICTION: CATEGORY + SAME OBJECT RECOGNITION
@app.route("/api/predict", methods=["POST"])
def predict_objects():

    if "image" not in request.files:
        return jsonify({"error": "Image is required."}), 400

    file = request.files["image"]

    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Unsupported image format."}), 400

    try:

        image_path = save_upload(file)

        extracted = segmenter.extract_objects(
            image_path,
            EXTRACTED_FOLDER / "predictions"
        )

        predictions = []

        for obj in extracted:

            embedding = feature_extractor.extract_features(
                obj["image_path"]
            )

            same_object = recognizer.find_same_object(
                embedding,
                obj["name"]
            )

            predictions.append({
                "name": obj["name"],
                "category_confidence": obj["confidence"],
                "same_object": same_object,
                "image_url": image_url(obj["image_path"]),
                "box": obj["box"]
            })

        return jsonify({
            "success": True,
            "total_objects": len(predictions),
            "predictions": predictions
        })

    except Exception as error:

        app.logger.exception("Prediction failed")

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "error": "Image size must not exceed 20 MB."
    }), 413


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
