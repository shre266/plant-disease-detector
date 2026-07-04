import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json

# Page config
st.set_page_config(page_title="Plant Disease Detector", 
                   page_icon="🌿", layout="centered")

# Treatment suggestions
TREATMENTS = {
    'healthy': "✅ Your plant looks healthy! Keep up the good care.",
    'Apple_scab': "🍎 Apply fungicides containing captan. Remove infected leaves.",
    'Black_rot': "🔴 Prune infected areas, apply copper-based fungicide.",
    'Cedar_apple_rust': "🍂 Apply fungicide in spring. Remove nearby juniper trees.",
    'Powdery_mildew': "⚪ Apply sulfur-based fungicide. Avoid overhead watering.",
    'Cercospora': "🌽 Apply fungicide, rotate crops next season.",
    'Common_rust': "🦠 Apply fungicide early. Plant resistant varieties.",
    'Northern_Leaf_Blight': "🌿 Apply fungicide, remove crop debris after harvest.",
    'Bacterial_spot': "💊 Apply copper-based bactericide. Avoid working with wet plants.",
    'Early_blight': "🍅 Apply fungicide, remove lower infected leaves.",
    'Late_blight': "⚠️ Apply fungicide immediately. This spreads fast!",
    'Leaf_Mold': "🍃 Improve ventilation, reduce humidity, apply fungicide.",
    'Septoria': "🔵 Remove infected leaves, apply fungicide.",
    'Spider_mites': "🕷️ Apply neem oil or insecticidal soap.",
    'Target_Spot': "🎯 Apply fungicide, remove infected leaves.",
    'Yellow_Leaf_Curl_Virus': "🟡 No cure — remove infected plant. Control whiteflies.",
    'Mosaic_virus': "🌈 No cure — remove infected plant. Control aphids.",
    'Leaf_scorch': "🍓 Ensure adequate watering, avoid excess fertilizer.",
    'Leaf_blight': "🍇 Apply fungicide, remove infected leaves.",
    'Haunglongbing': "🍊 No cure — remove infected tree to protect others.",
    'default': "💊 Consult a local agricultural expert for treatment advice."
}

def get_treatment(class_name):
    for key in TREATMENTS:
        if key.lower() in class_name.lower():
            return TREATMENTS[key]
    return TREATMENTS['default']

@st.cache_resource
def load_model():
    with open('models/class_names.json', 'r') as f:
        classes = json.load(f)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load('models/plant_disease_model.pth', 
                                      map_location='cpu'))
    model.eval()
    return model, classes

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# UI
st.title("🌿 Plant Disease Detector")
st.markdown("Upload a close-up photo of a plant leaf and the AI will detect any diseases.")
st.markdown("---")

uploaded_file = st.file_uploader("📸 Upload a leaf image", 
                                  type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Leaf", use_container_width=True)
    
    if st.button("🔍 Detect Disease", use_container_width=True):
        with st.spinner("Analyzing leaf..."):
            model, classes = load_model()
            img_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
            
            class_name = classes[predicted.item()]
            display_name = class_name.replace('___', ' — ').replace('_', ' ')
            confidence_pct = confidence.item() * 100
            
            st.markdown("---")
            st.markdown("### 🔬 Result:")
            
            if 'healthy' in class_name.lower():
                st.success(f"✅ **{display_name}**")
            else:
                st.error(f"⚠️ **{display_name}**")
            
            st.markdown(f"**Confidence:** {confidence_pct:.1f}%")
            st.progress(confidence.item())
            
            st.markdown("### 💊 Treatment Recommendation:")
            st.info(get_treatment(class_name))
            
            st.warning("⚠️ This is an AI prediction. Consult an agricultural expert for confirmation.")