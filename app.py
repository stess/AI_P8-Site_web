import logging
from flask import Flask, render_template, jsonify, request
import json
import os
import base64
import requests
from io import BytesIO
from PIL import Image

# Initialiser l'application Flask
app = Flask(__name__)

# Configurer le logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Charger le JSON et le transmettre à la page HTML


@app.route("/")
def home():
    json_path = os.path.join(app.static_folder, "images.json")
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            images_data = json.load(file)
        logging.info("Fichier JSON chargé avec succès.")
        return render_template("index.html", images=images_data)
    except Exception as e:
        logging.error(f"Erreur lors du chargement du fichier JSON : {e}")
        return jsonify({"error": "Failed to load JSON file"}), 500

# Helper pour encoder une image en base64


def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            logging.info(f"Encodage de l'image : {image_path}")
            return base64.b64encode(img_file.read()).decode('utf-8')
    except FileNotFoundError:
        logging.error(f"Fichier image introuvable : {image_path}")
        return None

# Route pour servir les images dynamiquement


@app.route("/get-images", methods=["POST"])
def get_images():
    data = request.json
    logging.info(f"Données reçues dans la requête : {data}")

    selected_image = data.get("selected_image", "")
    if not selected_image:
        logging.warning("Aucune image sélectionnée.")
        return jsonify({"error": "No image selected"}), 400

    # Construire les noms des fichiers
    left_img_path = os.path.join(
        app.static_folder, "images", f"{selected_image}_leftImg8bit.png")
    label_img_path = os.path.join(
        app.static_folder, "images", f"{selected_image}_gtFine_labelIds.png")

    # Encoder les images en base64
    left_img_base64 = encode_image_to_base64(left_img_path)
    label_img_base64 = encode_image_to_base64(label_img_path)

    if not left_img_base64 or not label_img_base64:
        logging.error(
            "L'une des images est introuvable ou n'a pas pu être encodée.")
        return jsonify({"error": "Images not found"}), 404

    # Appeler l'API externe pour obtenir l'image prédite
    predict_url = "http://localhost:5000/predict"
    try:
        logging.info(f"Envoi de la requête à l'API externe : {predict_url}")
        response = requests.post(predict_url, json={"image": left_img_base64})
        logging.info(f"Réponse de l'API externe : {response.status_code}")
        if response.status_code == 200:
            predicted_image_base64 = response.json().get("predicted_image", "")
            logging.info("Image prédite reçue avec succès.")
        else:
            predicted_image_base64 = ""
            logging.warning(
                f"Erreur dans la réponse de l'API externe : {response.status_code}")
    except requests.RequestException as e:
        logging.error(f"Échec de la requête à l'API externe : {e}")
        return jsonify({"error": f"Failed to fetch predicted image: {e}"}), 500

    return jsonify({
        "image1": left_img_base64,
        "image2": label_img_base64,
        "image3": predicted_image_base64  # Image prédite encodée en base64
    })


@app.route("/test-json")
def test_json():
    json_path = os.path.join(app.static_folder, "images.json")
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            logging.info("Test JSON : Chargement réussi.")
        return jsonify(data)
    except Exception as e:
        logging.error(f"Erreur lors du chargement du fichier JSON : {e}")
        return jsonify({"error": "Failed to load JSON file"}), 500


if __name__ == "__main__":
    logging.info("Lancement de l'application Flask...")
    app.run()
