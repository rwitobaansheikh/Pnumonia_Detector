import os
import torch
import torch.nn.functional as F
import librosa
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torchvision.transforms as T
from model import PneumoniaClassifierModel # Your custom model class
import tempfile

app = Flask(__name__)
CORS(app)

# --- 1. CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "Webapp/best_medical_model.pth"

# Setup X-ray transforms (Exact same as training)
img_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the model into memory once
model = PneumoniaClassifierModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# --- 2. PROCESSING FUNCTIONS ---

def process_audio(path):
    # Matches your Dataset audio_to_spec exactly
    y, sr = librosa.load(path, duration=3.0)
    if len(y) < sr * 3:
        y = np.pad(y, (0, int(sr * 3 - len(y))))
    
    ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    ps_db = librosa.power_to_db(ps, ref=np.max)
    
    # Normalize 0-1 as your training script did
    ps_db = (ps_db - ps_db.min()) / (ps_db.max() - ps_db.min())
    
    # Resize to 128x128 grayscale and convert to tensor
    img = Image.fromarray((ps_db * 255).astype(np.uint8)).resize((128, 128))
    spec_tensor = T.ToTensor()(img).unsqueeze(0).to(device)
    return spec_tensor

# --- 3. FLASK ROUTES ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'xray' not in request.files or 'audio' not in request.files:
            return jsonify({"error": "Missing files"}), 400
        
        xray_file = request.files['xray']
        audio_file = request.files['audio']

        # 1. Process X-ray (Keep as is)
        image = Image.open(xray_file).convert('RGB')
        image_tensor = img_transform(image).unsqueeze(0).to(device)

        # 2. Process Audio using a System Temp File
        # This saves to /tmp/ which Live Server DOES NOT watch
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio_file.save(tmp.name)
            spec_tensor = process_audio(tmp.name)
            temp_path = tmp.name # Store path to delete later

        # 3. Model Inference
        with torch.no_grad():
            logits = model(image_tensor, spec_tensor)
            probabilities = F.softmax(logits, dim=1)
            
        # Clean up the temp file after processing
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            "healthy": round(probabilities[0][0].item() * 100, 2),
            "pneumonia": round(probabilities[0][1].item() * 100, 2),
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)