# PneumoScan AI - Pneumonia Detector Application

A multimodal deep learning application that detects pneumonia using both chest X-ray images and respiratory audio recordings. The application combines computer vision and audio processing to provide accurate pneumonia risk assessment.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [System Components](#system-components)
4. [Installation & Setup](#installation--setup)
5. [Usage](#usage)
6. [Deployment](#deployment)
7. [CI/CD Pipeline](#cicd-pipeline)
8. [Technical Details](#technical-details)
9. [Live Application](#live-application)

---

## Overview

**PneumoScan AI** is a web-based diagnostic assistant that leverages a multimodal neural network to analyze patient data:

- **X-ray Images**: Visual chest imaging for pneumonia indicators
- **Audio Recordings**: Respiratory audio for cough and breath sound analysis

The system processes both inputs through separate neural network branches and combines the learned features to make a pneumonia risk prediction.

**Output**: Probability scores for:
- Healthy classification
- Pneumonia classification

---

## Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────┐
│           Frontend (HTML/CSS/JavaScript)                │
│                                                         │
│  - File upload interface (X-ray + Audio)                │
│  - Real-time preview display                            │
│  - Result visualization with confidence scores          │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP POST /predict
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Backend (Flask REST API)                        │
│                                                         │
│  - File reception & validation                          │
│  - X-ray preprocessing (normalization, resizing)        │
│  - Audio preprocessing (spectrogram conversion)         │
│  - Model inference coordination                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│      Multimodal Neural Network (PyTorch)                │
│                                                         │
│  ┌──────────────────┐        ┌──────────────────┐       │
│  │  Vision Branch   │        │  Audio Branch    │       │
│  │  (X-ray input)   │        │  (Spectrogram)   │       │
│  │                  │        │                  │       │
│  │  4× Conv Blocks  │        │  3× Conv Blocks  │       │
│  │  → 256 features  │        │  → 128 features  │       │
│  └────────┬─────────┘        └────────┬─────────┘       │
│           │                           │                 │
│           └─────────┬─────────────────┘                 │
│                     │ Concatenate                       │
│                     ▼                                   │
│           ┌─────────────────────┐                       │
│           │   Classification    │                       │
│           │   Head (2 classes)  │                       │
│           │ (Healthy/Pneumonia) │                       │
│           └─────────────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User uploads** X-ray image and audio file via web interface
2. **Frontend** sends files to Flask backend via POST request
3. **Backend processes**:
   - X-ray: Resize to 224×224, normalize with ImageNet statistics
   - Audio: Load as waveform, generate mel-spectrogram, normalize to 128×128
4. **Model inference**:
   - Vision branch processes X-ray → 256-dim feature vector
   - Audio branch processes spectrogram → 128-dim feature vector
   - Concatenate features (384 dimensions total)
   - Classification head outputs 2 class logits
5. **Softmax** converts logits to probabilities
6. **Results** returned to frontend as JSON with confidence percentages

---

## System Components

### 1. Frontend (`index.html`, `script/script.js`, `stylesheets/style.css`)

**Purpose**: User interface for file uploads and result display

**Features**:
- Dual file input (X-ray image + audio recording)
- Live preview of uploaded files
- Loading spinner during inference
- Color-coded results (red for high pneumonia risk, green for healthy)
- Reset button for new predictions

**Key JavaScript Functions**:
- `getPrediction()`: Handles form submission and API communication
- Image/audio preview event listeners
- Dynamic UI state management (loading, success, error states)

### 2. Backend (`app.py`)

**Purpose**: REST API server handling model inference

**Core Routes**:

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Serves index.html |
| `/<path>` | GET | Static file serving (CSS, JS) |
| `/predict` | POST | Main inference endpoint |

**Key Functions**:

- **`process_audio(path)`**: 
  - Loads audio file using librosa (3-second duration)
  - Generates mel-spectrogram (128 mel bands)
  - Converts to power DB scale
  - Normalizes to 0-1 range
  - Resizes to 128×128 grayscale tensor

- **`/predict` route**:
  - Validates input files
  - Processes X-ray image with torchvision transforms
  - Processes audio to spectrogram
  - Runs model inference
  - Returns JSON with probabilities

**Configuration**:
- Device: CUDA if available, otherwise CPU
- Model path: `best_medical_model.pth`
- Image transform: Standard ImageNet normalization

### 3. Model (`model.py`)

**Architecture**: Multimodal Convolutional Neural Network

**Vision Branch** (X-ray processor):
```
Input: 3×224×224 (RGB image)
↓
4× Conv Blocks [Conv2d → BatchNorm → ReLU → MaxPool]
  - Conv1: 3→32 channels
  - Conv2: 32→64 channels
  - Conv3: 64→128 channels
  - Conv4: 128→256 channels
↓
Adaptive Avg Pool → Flatten
↓
Output: 256-dimensional feature vector
```

**Audio Branch** (Spectrogram processor):
```
Input: 1×128×128 (grayscale spectrogram)
↓
3× Conv Blocks
  - Conv1: 1→32 channels
  - Conv2: 32→64 channels
  - Conv3: 64→128 channels
↓
Adaptive Avg Pool → Flatten
↓
Output: 128-dimensional feature vector
```

**Classification Head**:
```
Concatenated Features: 384 dimensions
↓
Dense(384→128) → ReLU → Dropout(0.5) → Dense(128→2)
↓
Output: 2 class logits (Healthy/Pneumonia)
↓
Softmax → Probability distribution
```

**Model Parameters**:
- Vision branch: ~256K parameters
- Audio branch: ~80K parameters
- Classifier: ~50K parameters
- **Total: ~386K parameters**

### 4. Dependencies (`requirements.txt`)

```
torch              # Deep learning framework
torchvision        # Image transforms and pretrained models
flask              # Web framework
flask-cors         # Cross-Origin Resource Sharing
librosa            # Audio processing
numpy              # Numerical computing
Pillow             # Image processing
```

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- pip or conda
- CUDA-compatible GPU (optional, CPU works fine)

### Local Setup

1. **Clone/Navigate to project directory**:
   ```bash
   cd /home/rwitobaansheikh/Web_apps/Pneumonia_detector/Webapp
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model file exists**:
   - Ensure `best_medical_model.pth` is in the project root
   - File size should be ~3-5 MB

5. **Run the application**:
   ```bash
   python app.py
   ```

6. **Access the web interface**:
   - Open browser to `http://localhost:5000`
   - Upload X-ray image and audio file
   - Click "Analyze Patient Data"

### Production Access

The application is deployed and accessible at: **https://pneumoniadetector.rwitobaansheikh.com/**

---

## Usage

### User Workflow

1. **Upload X-ray**:
   - Click "X-Ray Image" input
   - Select chest X-ray image file (PNG, JPG, etc.)
   - Preview appears automatically

2. **Upload Audio**:
   - Click "Audio Recording" input
   - Select respiratory audio file (WAV, MP3, etc.)
   - Audio player appears for preview

3. **Run Analysis**:
   - Click "Analyze Patient Data" button
   - Loading spinner appears (5-15 seconds depending on hardware)
   - Results display with probabilities

4. **Interpret Results**:
   - **Green text**: Healthy probability > 50% indicates low pneumonia risk
   - **Red text**: Pneumonia probability > 50% indicates high pneumonia risk
   - Both percentages sum to 100%

5. **New Prediction**:
   - Click "New Prediction" to reset form
   - Upload different files for another analysis

### Example Inference Request

```bash
curl -X POST http://localhost:5000/predict \
  -F "xray=@chest_xray.jpg" \
  -F "audio=@respiratory_audio.wav"
```

**Expected Response**:
```json
{
  "healthy": 85.32,
  "pneumonia": 14.68,
  "status": "success"
}
```

---

## Deployment

### Docker Deployment

The application includes a Dockerfile for containerized deployment.

**Build Docker image**:
```bash
docker build -t pneumonia-detector:latest .
```

**Run container**:
```bash
docker run -p 5000:5000 pneumonia-detector:latest
```

**Dockerfile Features**:
- Based on `python:3.10-slim` for minimal size
- Installs system dependencies for audio processing (libsndfile1, ffmpeg)
- Multi-stage build optimization (caches pip dependencies)
- Exposes port 5000 for Flask server

### AWS ECS Deployment

**Task Definition** (`task-def.json`):
- **Container**: pneumonia-detector (ECR image)
- **Memory**: 2048 MB
- **vCPU**: 512 (0.5 CPU)
- **Network Mode**: awsvpc (AWS VPC networking)
- **Launch Type**: Fargate (serverless)
- **Port Mapping**: 5000 → 5000

**Deployment Steps**:
1. Push Docker image to ECR:
   ```bash
   docker tag pneumonia-detector:latest \
     public.ecr.aws/u6o5d5r2/pneumonia-detector:latest
   docker push public.ecr.aws/u6o5d5r2/pneumonia-detector:latest
   ```

2. Register task definition:
   ```bash
   aws ecs register-task-definition --cli-input-json file://task-def.json
   ```

3. Create Fargate service and run tasks

### Production Deployment with Cloudflare & IONOS

**Domain Configuration**:
- **Domain**: rwitobaansheikh.com (registered on IONOS)
- **Subdomain**: pneumoniadetector.rwitobaansheikh.com
- **DNS Provider**: Cloudflare
- **SSL/TLS Certificate**: Managed by Cloudflare (Full encryption mode)
- **Live URL**: https://pneumoniadetector.rwitobaansheikh.com/

**Cloudflare Setup**:
1. Domain nameservers updated to point to Cloudflare
2. DNS records configured for pneumoniadetector subdomain
3. SSL/TLS encryption enabled (Full mode)
4. DDoS protection and caching enabled
5. Auto-renewal of HTTPS certificates

**IONOS Integration**:
- Domain registered and managed on IONOS platform
- Nameserver management delegated to Cloudflare for advanced features

---

## CI/CD Pipeline

The application uses **GitHub Actions** for automated deployment and continuous integration.

### Workflow Overview

```
Git Push to Repository
    ↓
GitHub Actions Triggered
    ↓
Run Tests & Build Docker Image
    ↓
Push to Container Registry
    ↓
Deploy to Production Server
    ↓
Update https://pneumoniadetector.rwitobaansheikh.com/
    ↓
Changes Live (within 2-5 minutes)
```

### Automatic Deployment Process

**Trigger**: Every `git push` to the repository automatically initiates the CI/CD pipeline

**Pipeline Steps**:
1. **Code Checkout**: GitHub Actions clones the latest repository code
2. **Build**: Creates Docker image with dependencies and model
3. **Push**: Uploads image to container registry
4. **Deploy**: Updates running container on production server
5. **Verify**: Health check to ensure application is responsive

**Deployment Result**:
- Changes reflected on https://pneumoniadetector.rwitobaansheikh.com/ within 2-5 minutes
- Automatic rollback on deployment failure
- Zero-downtime deployments

### GitHub Actions Configuration

**Typical Workflow File** (`.github/workflows/deploy.yml`):
```yaml
name: Deploy to Production

on:
  push:
    branches:
      - main
      - master

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: docker build -t pneumonia-detector:latest .
      
      - name: Push to registry
        run: |
          echo ${{ secrets.REGISTRY_PASSWORD }} | docker login -u ${{ secrets.REGISTRY_USER }} --password-stdin
          docker tag pneumonia-detector:latest ${{ secrets.REGISTRY_URL }}/pneumonia-detector:latest
          docker push ${{ secrets.REGISTRY_URL }}/pneumonia-detector:latest
      
      - name: Deploy to production
        run: |
          ssh ${{ secrets.DEPLOY_USER }}@${{ secrets.DEPLOY_HOST }} << 'EOF'
          cd /home/rwitobaansheikh/Web_apps/Pneumonia_detector/Webapp
          docker-compose pull
          docker-compose up -d
          EOF
```

### Secrets Management

GitHub Secrets used in CI/CD:
- `REGISTRY_URL`: Container registry URL
- `REGISTRY_USER`: Registry authentication user
- `REGISTRY_PASSWORD`: Registry authentication password
- `DEPLOY_USER`: SSH user for production server
- `DEPLOY_HOST`: Production server hostname/IP

### Development Workflow

1. **Local Development**:
   ```bash
   git clone https://github.com/rwitobaansheikh/Pneumonia_detector.git
   cd Webapp
   # Make changes to code
   ```

2. **Test Locally**:
   ```bash
   python app.py
   # Test on http://localhost:5000
   ```

3. **Commit & Push**:
   ```bash
   git add .
   git commit -m "Add new feature or fix"
   git push origin main
   ```

4. **Automatic Deployment**:
   - GitHub Actions automatically starts
   - Code builds and deploys
   - Changes live at https://pneumoniadetector.rwitobaansheikh.com/

5. **Monitor Deployment**:
   - Check GitHub Actions tab for workflow status
   - View deployment logs
   - Rollback if needed

---

## Technical Details

### Image Processing Pipeline

**Input**: Chest X-ray image (any size)

```
Original Image
    ↓
Resize to 224×224
    ↓
Convert to RGB tensor
    ↓
Normalize using ImageNet statistics:
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
    ↓
Add batch dimension: 1×3×224×224
    ↓
Move to device (GPU/CPU)
```

### Audio Processing Pipeline

**Input**: Audio file (any format, any sample rate)

```
Audio File
    ↓
Load with librosa (3-second duration max)
    ↓
Pad to 3 seconds if shorter
    ↓
Generate mel-spectrogram:
  - Sample rate: native (auto-detected)
  - Mel bands: 128
    ↓
Convert to power dB scale
    ↓
Normalize: (spec - min) / (max - min)
    ↓
Resize spectrogram to 128×128
    ↓
Convert to grayscale tensor
    ↓
Add batch dimension: 1×1×128×128
    ↓
Move to device (GPU/CPU)
```

### Model Inference Process

1. **Forward Pass**:
   - Vision branch processes X-ray → 256 features
   - Audio branch processes spectrogram → 128 features
   - Features concatenated → 384 dimensions

2. **Classification**:
   - Dense layer (384→128) with ReLU activation
   - Dropout (50% during training, no effect during inference)
   - Dense layer (128→2) outputs logits

3. **Probability Calculation**:
   - Apply softmax: $p_i = \frac{e^{z_i}}{\sum_{j=1}^{2} e^{z_j}}$
   - Healthy probability: percentage for class 0
   - Pneumonia probability: percentage for class 1

### Performance Characteristics

- **Inference Time**: 2-5 seconds (CPU), <1 second (GPU)
- **Model Size**: ~3.8 MB
- **Memory Usage**: ~500 MB (PyTorch + dependencies)
- **GPU Memory**: ~1 GB when using CUDA

### Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| "Missing files" | X-ray or audio not uploaded | Upload both files |
| "Server error" | Flask not running | Start Flask server on port 5000 |
| Audio processing error | Unsupported audio format | Use WAV, MP3, or standard formats |
| Image processing error | Invalid image file | Use standard image formats (JPG, PNG) |

---

## Project Structure

```
Pneumonia_detector/Webapp/
├── README.md                    # This file
├── app.py                       # Flask backend
├── model.py                     # PyTorch model architecture
├── best_medical_model.pth       # Pre-trained model weights
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker containerization
├── task-def.json               # AWS ECS task definition
├── index.html                  # Frontend HTML
├── script/
│   └── script.js              # Frontend JavaScript
└── stylesheets/
    └── style.css              # Frontend CSS
```

---

## Future Enhancements

- Add authentication for medical professionals
- Implement result history/patient records
- Add model explainability (CAM, attention maps)
- Expand to multiclass classification (normal, pneumonia, tuberculosis, etc.)
- Real-time audio recording directly in browser
- Integration with medical record systems (DICOM support)
- Batch processing for multiple patient files
- Model versioning and A/B testing

---

## Troubleshooting

### Flask server won't start
- Check if port 5000 is already in use: `lsof -i :5000`
- Kill existing process: `kill -9 <PID>`
- Restart Flask: `python app.py`

### Model file not found
- Ensure `best_medical_model.pth` is in the project root
- Check file permissions: `ls -la best_medical_model.pth`

### Audio processing fails
- Verify audio file is valid: `file audio.wav`
- Try converting to WAV: `ffmpeg -i audio.mp3 audio.wav`

### Out of memory errors
- Reduce batch size (currently processes 1 file at a time)
- Close other applications
- Use CPU instead of GPU: Edit device line in `app.py`

---

## Live Application

**Access the deployed application**: https://pneumoniadetector.rwitobaansheikh.com/

The application is running 24/7 on a production server with automatic deployments via GitHub Actions CI/CD pipeline.

**Features**:
- HTTPS secured with Cloudflare certificates
- Automatic code deployment on every git push
- Zero-downtime updates
- Global CDN caching via Cloudflare
- DDoS protection
- Uptime monitoring

---

## Contact & Support

For issues, questions, or improvements, please refer to the project documentation or contact me through the following platforms.

- GitHub: https://github.com/rwitobaansheikh
- LinkedIn: https://www.linkedin.com/in/rwitobaansheikh/
- Email: rwitobaansheikh@gmail.com

---

**Last Updated**: February 21, 2026  
**Status**: Production Ready  
**Deployment**: https://pneumoniadetector.rwitobaansheikh.com/  
**CI/CD**: GitHub Actions (Auto-deploy on push)
