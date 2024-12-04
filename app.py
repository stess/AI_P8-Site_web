from flask import Flask, render_template, jsonify, request
import json
import os
import base64
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Initialiser l'application Flask
app = Flask(__name__)


# Fonction pour coloriser une image avec la palette tab10


def apply_tab10_palette(image_array):
    """
    Applique la palette tab10 sur une image contenant des indices de classes.
    Args:
        image_array (np.ndarray): Image avec indices des classes.
    Returns:
        np.ndarray: Image colorisée.
    """
    cmap = plt.get_cmap('tab10')  # Palette tab10
    normalized = image_array / image_array.max()  # Normalisation entre 0 et 1
    colored_image = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)  # RGB
    return colored_image

# Fonction pour coloriser une image avec la palette tab20


def apply_tab20_palette(image_array):
    """
    Applique la palette tab20 sur une image contenant des indices de classes.
    Args:
        image_array (np.ndarray): Image avec indices des classes.
    Returns:
        np.ndarray: Image colorisée.
    """
    cmap = plt.get_cmap('tab20')  # Palette tab20
    normalized = image_array / image_array.max()  # Normalisation entre 0 et 1
    colored_image = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)  # RGB
    return colored_image


@app.route("/")
def home():
    json_path = os.path.join(app.static_folder, "images.json")
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            images_data = json.load(file)
        return render_template("index.html", images=images_data)
    except Exception as e:
        return jsonify({"error": "Failed to load JSON file"}), 500

# Helper pour encoder une image en base64


def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except FileNotFoundError:
        return None

# Fonction pour coloriser une image avec la palette tab10


@app.route("/get-images", methods=["POST"])
def get_images():
    data = request.json

    selected_image = data.get("selected_image", "")
    if not selected_image:
        return jsonify({"error": "No image selected"}), 400

    # Construire les noms des fichiers
    left_img_path = os.path.join(
        app.static_folder, "images", f"{selected_image}_leftImg8bit.png")
    label_img_path = os.path.join(
        app.static_folder, "images", f"{selected_image}_gtFine_labelIds.png")

    # Charger et encoder l'image 1 (ne doit pas être modifiée)
    left_img_base64 = encode_image_to_base64(left_img_path)

    # Charger l'image 2 (label) et appliquer la palette tab10
    try:
        label_image = Image.open(label_img_path).convert(
            "L")  # Charger en niveaux de gris
        label_array = np.array(label_image)
        label_colored = apply_tab20_palette(label_array)  # Appliquer tab10
        label_colored_pil = Image.fromarray(label_colored)  # Convertir en PIL
        buffer = BytesIO()
        # Sauvegarder dans un buffer
        label_colored_pil.save(buffer, format="PNG")
        label_img_base64 = base64.b64encode(
            buffer.getvalue()).decode('utf-8')  # Encoder
    except Exception as e:
        return jsonify({"error": "Failed to process image 2"}), 500

    # Appeler l'API externe pour obtenir l'image prédite (image 3)
    predict_url = "http://localhost:5001/predict"
    try:
        response = requests.post(predict_url, json={"image": left_img_base64})
        if response.status_code == 200:
            predicted_image_base64 = response.json().get("predicted_image", "")
            if predicted_image_base64:
                # Décoder l'image 3, appliquer la palette et ré-encoder
                predicted_image_data = base64.b64decode(predicted_image_base64)
                predicted_image = Image.open(
                    BytesIO(predicted_image_data)).convert("L")
                predicted_array = np.array(predicted_image)
                predicted_colored = apply_tab10_palette(
                    predicted_array)  # Appliquer tab10
                predicted_colored_pil = Image.fromarray(predicted_colored)
                buffer = BytesIO()
                predicted_colored_pil.save(buffer, format="PNG")
                predicted_image_base64 = base64.b64encode(
                    buffer.getvalue()).decode('utf-8')
        else:
            predicted_image_base64 = ""
    except requests.RequestException as e:
        return jsonify({"error": f"Failed to fetch predicted image: {e}"}), 500

    return jsonify({
        "image1": left_img_base64,
        "image2": label_img_base64,  # Image 2 colorisée
        "image3": predicted_image_base64  # Image 3 colorisée
    })


@app.route("/test-json")
def test_json():
    json_path = os.path.join(app.static_folder, "images.json")
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": "Failed to load JSON file"}), 500


if __name__ == "__main__":
    app.run()
