// 1. Image Preview Logic
document.getElementById('xrayInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const preview = document.getElementById('imagePreview');
        preview.src = URL.createObjectURL(file);
        preview.style.display = 'block';
    }
});

// 2. Audio Preview Logic
document.getElementById('audioInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const preview = document.getElementById('audioPreview');
        preview.src = URL.createObjectURL(file);
        preview.style.display = 'block';
    }
});

// 3. Main Prediction Logic
async function getPrediction(e) {
    e.preventDefault();
    
    const xray = document.getElementById('xrayInput').files[0];
    const audio = document.getElementById('audioInput').files[0];
    
    if(!xray || !audio) return alert("Please upload both files!");

    const btn = document.getElementById('submit');
    btn.innerText = "Processing Diagnostic...";
    btn.disabled = true;

    const formData = new FormData();
    formData.append('xray', xray);
    formData.append('audio', audio);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        // Display Results
        document.getElementById('resultDisplay').style.display = 'block';
        
        const pneu = document.getElementById('pneuResult');
        pneu.innerText = `Pneumonia Risk: ${data.pneumonia}%`;
        pneu.style.color = data.pneumonia > 50 ? "#e53e3e" : "#38a169"; // Red if high risk, Green if low
        
        document.getElementById('healthyResult').innerText = `Healthy Probability: ${data.healthy}%`;

    } catch (err) {
        console.error(err);
        alert("Server error. Ensure Flask is running on port 5000.");
    } finally {
        btn.innerText = "Analyze Patient Data";
        btn.disabled = false;
    }
}

document.getElementById('resetBtn').addEventListener('click', function() {
    // 1. Reset the actual form (clears file inputs)
    document.getElementById('uploadForm').reset();

    // 2. Hide and clear Previews
    const imagePreview = document.getElementById('imagePreview');
    const audioPreview = document.getElementById('audioPreview');
    
    imagePreview.src = "";
    imagePreview.style.display = "none";
    
    audioPreview.src = "";
    audioPreview.pause(); // Stop audio if playing
    audioPreview.style.display = "none";

    // 3. Hide the AI results
    document.getElementById('resultDisplay').style.display = "none";
    
    console.log("Form cleared for new patient data.");
});

document.getElementById('uploadForm').addEventListener('submit', getPrediction);