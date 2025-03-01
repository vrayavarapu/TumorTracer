let model;
let classLabels = [];

async function loadModel() {
    try {
        console.log("Loading model...");
        model = await tf.loadGraphModel('model/model.json');
        console.log("✅ Model loaded successfully!");

        // Load metadata.json for class labels
        const response = await fetch('model/metadata.json');
        const metadata = await response.json();
        classLabels = metadata.labels || [];
        console.log("✅ Class labels loaded:", classLabels);
    } catch (error) {
        console.error("❌ Error loading model:", error);
        document.getElementById("prediction").innerText = "Error: Model not loaded.";
    }
}

async function predictImage() {
    if (!model) {
        console.error("❌ Model is not loaded yet!");
        return;
    }

    let imageElement = document.getElementById("imagePreview");
    if (!imageElement) {
        console.error("❌ No image found for prediction.");
        return;
    }

    let tensor = tf.browser.fromPixels(imageElement)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(tf.scalar(255))
        .expandDims();

    let predictions = await model.predict(tensor).data();
    console.log("Predictions:", predictions);

    let maxIndex = predictions.indexOf(Math.max(...predictions));
    let predictedClass = classLabels[maxIndex] || `Class ${maxIndex}`;

    document.getElementById("prediction").innerText = `Prediction: ${predictedClass} (${(predictions[maxIndex] * 100).toFixed(2)}%)`;
}

function handleImageUpload(event) {
    let reader = new FileReader();
    reader.onload = function () {
        let img = document.getElementById("imagePreview");
        img.src = reader.result;
        img.hidden = false;
        img.onload = function () {
            predictImage();
        };
    };
    reader.readAsDataURL(event.target.files[0]);
}

document.getElementById("imageUpload").addEventListener("change", handleImageUpload);
document.addEventListener("DOMContentLoaded", loadModel);
