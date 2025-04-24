import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
import time
import requests
from datetime import datetime
import json
from ultralytics import YOLO
import tensorflow as tf
from openai import OpenAI
from PIL import Image
import librosa
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image, ImageDraw
import pandas as pd
import matplotlib.pyplot as plt

def render_app_documentation():
    
    st.markdown("""
    <style>
    .info-box {
        background-color: #2c3e50;
        border-left: 5px solid #3498db;
        padding: 15px;
        margin: 20px 0;
        border-radius: 5px;
    }
    
    .info-box h3 {
        margin-top: 0;
        color: #ecf0f1;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin: 25px 0;
    }
    
    .feature-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .audio-card {
        background: linear-gradient(to bottom right, #103a24, #1a5336);
    }
    
    .audio-card h3 {
        color: #4ade80;
        margin-top: 0;
    }
    
    .image-card {
        background: linear-gradient(to bottom right, #0c4a6e, #075985);
    }
    
    .image-card h3 {
        color: #67e8f9;
        margin-top: 0;
    }
    
    .video-card {
        background: linear-gradient(to bottom right, #4a1d3a, #6b2952);
    }
    
    .video-card h3 {
        color: #fb7185;
        margin-top: 0;
    }
    
    .ai-card {
        background: linear-gradient(to bottom right, #2e1065, #4c1d95);
    }
    
    .ai-card h3 {
        color: #c4b5fd;
        margin-top: 0;
    }
    
    .step-container {
        margin-top: 40px;
    }
    
    .step-box {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        border-left: 4px solid #3498db;
    }
    
    .step-box h4 {
        margin-top: 0;
        color: #e2e8f0;
    }
    
    .tech-tag {
        display: inline-block;
        background-color: #334155;
        color: #e2e8f0;
        padding: 5px 10px;
        border-radius: 15px;
        margin-right: 8px;
        margin-bottom: 8px;
        font-size: 0.85em;
    }
    
    p {
        color: #cbd5e1;
    }
    
    h2, h3, h4 {
        color: #f1f5f9;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>About This Application</h3>
        <p>This application analyzes audio, image, and video files to determine if the audio matches the visual content. 
        It uses advanced machine learning models to detect objects, classify sounds, and provide intelligent verification results.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## ✨ Key Features")
    
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card audio-card">
            <h3>🔊 Audio Analysis</h3>
            <p>Classifies audio into categories and subcategories using MFCC and mel spectrogram features</p>
        </div>
        <div class="feature-card image-card">
            <h3>🖼️ Image Analysis</h3>
            <p>Detects objects in images using the Imagga API with confidence scores</p>
        </div>
        <div class="feature-card video-card">
            <h3>🎥 Video Analysis</h3>
            <p>Identifies objects in videos using YOLOv8 object detection with frame-by-frame tracking</p>
        </div>
        <div class="feature-card ai-card">
            <h3>🧠 AI Interpretation</h3>
            <p>Compares audio and visual data to determine if they match using LLM-powered analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🔍 How It Works")
    
    st.markdown("""
            ### Step 1: Upload Files
            Upload your audio (.wav, .mp3, .ogg), image (.jpg, .png), and video (.mp4, .mov, .avi) files 
            in the "Upload Files" tab. Each file will be processed individually.

            ### Step 2: Processing
            The system extracts audio features, analyzes images for objects using Imagga's API, 
            and processes videos with YOLOv8 object detection.

        
            ### Step 3: AI Analysis
            An AI model compares the audio classification with detected visual objects to determine if 
            the audio reasonably matches what is seen in the image/video.
        
            ### Step 4: Results & Visualization
            View detailed analysis with interactive charts and visualizations in the "Results" tab. 
            Download a comprehensive report of the findings.

    """, unsafe_allow_html=True)
    
    st.markdown("## 🛠️ Technology Stack")
    
    st.markdown("""
    <div>
        <span class="tech-tag">Streamlit</span>
        <span class="tech-tag">TensorFlow</span>
        <span class="tech-tag">YOLOv8</span>
        <span class="tech-tag">Librosa</span>
        <span class="tech-tag">Plotly</span>
        <span class="tech-tag">OpenCV</span>
        <span class="tech-tag">Imagga API</span>
        <span class="tech-tag">LLM APIs</span>
        <span class="tech-tag">Pandas</span>
        <span class="tech-tag">NumPy</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🚀 Use Cases")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Content Verification
        - Detect audio-visual mismatches in media
        - Identify potentially misleading content
        - Verify authenticity of recordings
        """)
        
    with col2:
        st.markdown("""
        ### Research & Analysis
        - Study relationships between sounds and visuals
        - Analyze environmental audio recordings
        - Test audio recognition systems
        """)
    
    st.markdown("## 🏁 Getting Started")
    
    st.info("""
    **To begin using the application:**
    1. Navigate to the "Upload Files" tab
    2. Upload your audio, image, and/or video files
    3. Click "Process Uploaded Files"
    4. Go to the "Results" tab to view the detailed analysis
    """)
    

st.set_page_config(page_title="Audio-Visual Verification System", layout="wide")

IMAGGA_API_KEY = "acc_aeaca4a5cd61e4d"
IMAGGA_API_SECRET = "0890f560ffe4025210fe54cc56f6a63c"
OPENAI_API_KEY = "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"

openai_client = OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource
def load_yolo_model(model_path="yolov8n.pt"):
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Downloading...")
        model = YOLO("yolov8n.pt")  
        model.save(model_path)
    else:
        print(f"Loading model from {model_path}")
        model = YOLO(model_path)
    return model

@st.cache_resource
def load_audio_model():
    model = tf.keras.models.load_model('NasNet_Mobile/audio_classifier.h5')
    return model

def load_metadata():
    model_path = 'NasNet_Mobile'
    metadata_file = os.path.join(model_path, 'metadata.npy')
    
    if not os.path.exists(metadata_file):
        st.error(f"MetaData files not found in {model_path}. Please check the path.")
        return None
        
    metadata = np.load(metadata_file, allow_pickle=True).item()

    return metadata

def extract_features(file_path, params):
    try:
        audio, sr = librosa.load(file_path, sr=params['sample_rate'], duration=params['duration'])

        if len(audio) < params['sample_rate'] * params['duration']:
            padding = params['sample_rate'] * params['duration'] - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')

        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=params['sample_rate'],
            n_mfcc=params['n_mfcc'],
            n_fft=params['n_fft'],
            hop_length=params['hop_length']
        )

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=params['sample_rate'],
            n_mels=params['n_mels'],
            n_fft=params['n_fft'],
            hop_length=params['hop_length']
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
        mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / np.std(mel_spec_db)

        return {
            'mfccs': mfccs,
            'mel_spec': mel_spec_db,
            'audio': audio,
            'sr': sr
        }

    except Exception as e:
        st.error(f"Error extracting features from {file_path}: {e}")
        return None
    
def extract_audio_features(audio_file, sr=22050, duration=5):
    y, sr = librosa.load(audio_file, sr=sr, duration=duration)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    features = {
        'mfccs': mfccs.T.mean(axis=0),
        'chroma': chroma.T.mean(axis=0),
        'mel': mel.T.mean(axis=0),
        'contrast': contrast.T.mean(axis=0)
    }
    
    return features

def visualize_audio_features(features):
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Waveform", "Mel Spectrogram", "MFCCs", "Audio Power"),
        specs=[[{"type": "xy"}, {"type": "heatmap"}],
               [{"type": "heatmap"}, {"type": "xy"}]]
    )
    
    times = np.arange(len(features['audio'])) / features['sr']
    fig.add_trace(
        go.Scatter(x=times, y=features['audio'], mode='lines', name='Waveform'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Heatmap(
            z=features['mel_spec'],
            colorscale='Viridis',
            name='Mel Spectrogram'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Heatmap(
            z=features['mfccs'],
            colorscale='Inferno',
            name='MFCCs'
        ),
        row=2, col=1
    )
    
    hop_length = 512
    frame_length = 2048
    rms = librosa.feature.rms(y=features['audio'], frame_length=frame_length, hop_length=hop_length)[0]
    times_rms = np.arange(len(rms)) * hop_length / features['sr']
    fig.add_trace(
        go.Scatter(x=times_rms, y=rms, mode='lines', name='RMS Energy'),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600, width=900,
        title_text="Audio Feature Visualization"
    )
    
    return fig

def predict_with_filters(features, model, metadata, selected_categories=None, selected_subcategories=None):

    X_mfcc = np.array([features['mfccs']])[..., np.newaxis]
    
    category_mapping = metadata['category_mapping']
    subcategory_mapping = metadata['subcategory_mapping']
    category_to_subcategories = metadata['category_to_subcategories']
    
    reverse_category_mapping = {v: k for k, v in category_mapping.items()}
    reverse_subcategory_mapping = {v: k for k, v in subcategory_mapping.items()}
    
    filtered_category_indices = None
    if selected_categories and len(selected_categories) > 0:
        filtered_category_indices = [reverse_category_mapping[cat] for cat in selected_categories 
                                     if cat in reverse_category_mapping]
    
    filtered_subcategory_indices = None
    if selected_subcategories and len(selected_subcategories) > 0:
        filtered_subcategory_indices = [reverse_subcategory_mapping[subcat] for subcat in selected_subcategories 
                                       if subcat in reverse_subcategory_mapping]
    
    raw_predictions = model.predict(X_mfcc, verbose=0)
    
    category_preds = raw_predictions[0][0].copy() 
    subcategory_preds = raw_predictions[1][0].copy()  
    
    if filtered_category_indices is not None and len(filtered_category_indices) > 0:
        for i in range(len(category_preds)):
            if i not in filtered_category_indices:
                category_preds[i] = 0
        
        if np.sum(category_preds) > 0:
            category_preds = category_preds / np.sum(category_preds)
    
    pred_category_idx = np.argmax(category_preds)
    
    valid_subcategory_indices = category_to_subcategories[pred_category_idx]
    
    for j in range(len(subcategory_preds)):
        if j not in valid_subcategory_indices:
            subcategory_preds[j] = 0
    
    if filtered_subcategory_indices is not None and len(filtered_subcategory_indices) > 0:
        for j in range(len(subcategory_preds)):
            if j not in filtered_subcategory_indices:
                subcategory_preds[j] = 0
    
    if np.sum(subcategory_preds) > 0:
        subcategory_preds = subcategory_preds / np.sum(subcategory_preds)
    else:
        valid_indices = [idx for idx in valid_subcategory_indices if (filtered_subcategory_indices is None or 
                                                                      idx in filtered_subcategory_indices)]
        if valid_indices:
            for idx in valid_indices:
                subcategory_preds[idx] = 1.0 / len(valid_indices)
    
    subcategory_idx = np.argmax(subcategory_preds)

    top_categories = np.argsort(category_preds)[-3:][::-1]
    top_categories = [idx for idx in top_categories if category_preds[idx] > 0]
    
    top_subcategories = np.argsort(subcategory_preds)[-3:][::-1]
    top_subcategories = [idx for idx in top_subcategories if subcategory_preds[idx] > 0]
    
    result = {
        'category': category_mapping[pred_category_idx],
        'category_confidence': float(category_preds[pred_category_idx]),
        'subcategory': subcategory_mapping[subcategory_idx] if np.max(subcategory_preds) > 0 else "None",
        'subcategory_confidence': float(subcategory_preds[subcategory_idx]) if np.max(subcategory_preds) > 0 else 0.0,
        'top_categories': [(category_mapping[idx], float(category_preds[idx])) for idx in top_categories],
        'top_subcategories': [(subcategory_mapping[idx], float(subcategory_preds[idx])) for idx in top_subcategories],
        'features': features
    }
    
    return result

def visualize_audio_classifications(audio_result):
    # Create two subplots for categories and subcategories
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=("Top Categories", "Top Subcategories"),
                       specs=[[{"type": "bar"}, {"type": "bar"}]])
    
    # Categories visualization
    categories = [cat for cat, _ in audio_result['top_categories']]
    confidences = [conf * 100 for _, conf in audio_result['top_categories']]
    
    fig.add_trace(
        go.Bar(
            x=categories,
            y=confidences,
            marker_color=['rgb(26, 118, 255)' if cat == audio_result['category'] else 'lightblue' for cat in categories],
            text=[f"{conf:.1f}%" for conf in confidences],
            textposition="auto",
            name="Categories"
        ),
        row=1, col=1
    )
    
    # Subcategories visualization
    subcategories = [subcat for subcat, _ in audio_result['top_subcategories']]
    sub_confidences = [conf * 100 for _, conf in audio_result['top_subcategories']]
    
    fig.add_trace(
        go.Bar(
            x=subcategories,
            y=sub_confidences,
            marker_color=['rgb(26, 118, 255)' if subcat == audio_result['subcategory'] else 'lightblue' for subcat in subcategories],
            text=[f"{conf:.1f}%" for conf in sub_confidences],
            textposition="auto",
            name="Subcategories"
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Audio Classification Results",
        height=400,
        showlegend=False
    )
    
    return fig

def analyze_image_with_imagga(image_file):
    img_bytes = io.BytesIO()
    image_file.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()

    auth = (IMAGGA_API_KEY, IMAGGA_API_SECRET)
    files = {
        'image': ('image.jpg', img_bytes, 'image/jpeg')
    }

    response = requests.post(
        'https://api.imagga.com/v2/tags',
        files=files,
        auth=auth
    )

    if response.status_code != 200:
        return {"error": f"Imagga API error: {response.status_code}"}

    result = response.json()
    tags = result['result']['tags']

    objects = [{"tag": tag['tag']['en'], "confidence": tag['confidence']}
               for tag in tags if tag['confidence'] > 30]
    print(objects)
    return {"objects": objects}


def visualize_object_detection(objects, title="Object Detection Results"):
    # Filter out top objects by confidence for cleaner visualization
    top_objects = sorted(objects, key=lambda x: x['confidence'], reverse=True)[:10]
    
    # Extract tags and confidences
    tags = [obj['tag'] for obj in top_objects]
    confidences = [obj['confidence'] for obj in top_objects]
    
    # Create horizontal bar chart for better readability with long labels
    fig = go.Figure(go.Bar(
        y=tags,
        x=confidences if isinstance(confidences[0], float) else [conf for conf in confidences],
        orientation='h',
        marker_color='rgb(55, 83, 109)',
        text=[f"{conf:.1f}%" if isinstance(conf, float) else f"{conf:.1f}%" for conf in confidences],
        textposition="auto"
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Confidence (%)",
        yaxis_title="Object",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig

def draw_boxes_on_image(image_bytes, objects):
    """
    This function is a placeholder for drawing boxes on detected objects
    In a real implementation, you would need bounding box coordinates from the detection API
    """
    # For now, we'll just annotate the image with labels
    img = Image.open(io.BytesIO(image_bytes))
    draw = ImageDraw.Draw(img)
    
    # This is just a placeholder - in reality you'd need actual bounding box coordinates
    # from your object detection API
    
    return img

def analyze_video_with_yolo(video_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(video_file.read())
        temp_path = temp_file.name

    try:
        model = load_yolo_model()
        results = model.track(temp_path, save=False, stream=True)

        detected_objects = {}
        frame_count = 0
        frames_with_detections = []

        for r in results:
            if frame_count > 30:
                break

            frame_count += 1
            boxes = r.boxes.cpu().numpy()

            if len(boxes) > 0:
                frame = r.orig_img.copy()
                frames_with_detections.append((frame, boxes, r.names))

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = r.names[cls_id]

                if name in detected_objects:
                    detected_objects[name] = max(detected_objects[name], conf)
                else:
                    detected_objects[name] = conf

        objects = [{"tag": obj, "confidence": conf}
                   for obj, conf in detected_objects.items() if conf > 0.5]

        # 🧹 Wait and retry unlink to avoid PermissionError
        for _ in range(3):
            try:
                os.unlink(temp_path)
                break
            except PermissionError:
                time.sleep(1)

        print(f"Detected objects: {objects}")
        print(f"Frames with detections: {frames_with_detections}")
        
        return {
            "objects": objects,
            "frames_with_detections": frames_with_detections[:5]
        }

    except Exception as e:
        # Cleanup on error too
        for _ in range(3):
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                break
            except PermissionError:
                time.sleep(1)

        return {"error": f"YOLO processing error: {str(e)}"}

def visualize_frames_with_detections(frames_with_detections):

    if not frames_with_detections:
        return None
    
    fig, axes = plt.subplots(1, min(len(frames_with_detections), 5), figsize=(15, 5))
    if len(frames_with_detections) == 1:
        axes = [axes]  
    
    for i, (frame, boxes, names) in enumerate(frames_with_detections[:5]): 
        if i >= len(axes):
            break

        # Convert the BGR NumPy array to RGB using PIL
        image = Image.fromarray(frame[..., ::-1])  # Convert BGR to RGB
        axes[i].imshow(image)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = names[cls_id]

            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                fill=False, edgecolor='red', linewidth=2)
            axes[i].add_patch(rect)
            axes[i].text(x1, y1-10, f"{class_name}: {conf:.2f}",
                        color='white', fontsize=10,
                        bbox=dict(facecolor='red', alpha=0.5))

        axes[i].set_title(f"Frame {i+1}")
        axes[i].axis('off')

    plt.tight_layout()
    return fig

def create_verification_summary_visualization(audio_result, image_result, video_result, ai_interpretation):

    match_level = "Unknown"
    
    interpretation_text = ai_interpretation.lower()
    if "match" in interpretation_text:
        if any(phrase in interpretation_text for phrase in ["does not match", "doesn't match", "no match", "mismatch"]):
            match_level = "No Match"
        elif "partial match" in interpretation_text or "somewhat matches" in interpretation_text:
            match_level = "Partial Match"
        else:
            match_level = "Match"
    
    if match_level == "Match":
        gauge_value = 0.9
        color = "green"
    elif match_level == "Partial Match":
        gauge_value = 0.5
        color = "orange"
    elif match_level == "No Match":
        gauge_value = 0.1
        color = "red"
    else:
        gauge_value = 0.5
        color = "gray"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = gauge_value * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Audio-Visual Verification: {match_level}"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': gauge_value * 100
            }
        }
    ))
    
    fig.update_layout(height=400)
    
    return fig

def get_ai_interpretation(audio_result, image_result, video_result):
    
    import requests
    import os
    import json
    
    audio_data = f"Audio classification: {audio_result['subcategory']} ({audio_result['subcategory_confidence']*100:.2f}%) in category {audio_result['category']} ({audio_result['category_confidence']*100:.2f}%)"
    
    image_data = "Image contains: "
    if "objects" in image_result and image_result["objects"]:
        image_data += ", ".join([f"{obj['tag']} ({obj['confidence']*100:.2f}%)" for obj in image_result["objects"][:5]])
    else:
        image_data += "No objects detected or an error occurred."
    
    video_data = "Video contains: "
    if "objects" in video_result and video_result["objects"]:
        video_data += ", ".join([f"{obj['tag']} ({obj['confidence']*100:.2f}%)" for obj in video_result["objects"][:5]])
    else:
        video_data += "No objects detected or an error occurred."
    
    prompt = f"""
    Based on the following analysis results, determine if the audio source matches the visual evidence:
    
    {audio_data}
    
    {image_data}
    
    {video_data}
    
    Does the audio prediction match what is seen in the image/video?
    if the audio or video is not present, please mention that. and dont include it in the comparison.
    If yes, explain why the sound likely came from the detected object.
    If no, suggest what might have produced the sound instead (e.g., "dog sound was played through a speaker").
    Keep your answer concise and clear.
    """
    
    try:
        api_key = os.environ.get("GROQ_API_KEY", "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")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        payload = {
            "model": "llama3-8b-8192",  
            "messages": [
                {"role": "system", "content": "You are an AI assistant that analyzes audio and visual data to determine if they match."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 300
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return f"Unexpected response format: {json.dumps(result)}"
        else:
            return f"API error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"AI interpretation error: {str(e)}"
    
def get_parameters():
    return {
        'data_dir': 'DATASET',
        'sample_rate': 22050,
        'duration': 5,  
        'n_mfcc': 40,
        'n_mels': 128,
        'n_fft': 2048,
        'hop_length': 512,
        'batch_size': 32,
        'epochs': 20,
        'validation_split': 0.2,
        'random_state': 42,
        'num_test_samples': 10  
    }
def load_demo_files():
    
    demo_files = {
        "audio": {
            "cars_sound.wav": {"path": "demo_files/audio/car_119.wav", "description": "multiple cars in the street sound"},
            "elephant.wav": {"path": "demo_files/audio/elephant_1_part_13.wav", "description": "elephant sound trumpeting"}, 
        },
        "image": {
            "car_multiple.jpg": {"path": "demo_files/images/car_multiple.jpg", "description": "car multiple in the street image"},
            "cellphone.jpg": {"path": "demo_files/images/cellphone.jpg", "description": "mobile on a table image"},
            "elephant_nature.jpg": {"path": "demo_files/images/elephant_nature.jpg", "description": "elephant in nature image"},
        },
        "video": {
            "car_street.mp4": {"path": "demo_files/videos/segment_004.mp4", "description": "Car in the street video"},
            "elephant.mp4": {"path": "demo_files/videos/segment_040.mp4", "description": "Elephant walking video"},
            "mobile_table.mp4": {"path": "demo_files/videos/mobile_table.mp4", "description": "Mobile on table video"},
        }
    }
    return demo_files

def main():
    st.title("Audio-Visual Verification System")
    
    render_app_documentation()
    
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None
    if 'features' not in st.session_state:
        st.session_state.features = None
    if 'current_model' not in st.session_state:
        st.session_state.current_model = load_audio_model()
    if 'metadata' not in st.session_state:
        st.session_state.metadata = load_metadata()
    if 'audio_result' not in st.session_state:
        st.session_state.audio_result = None
    if 'image_result' not in st.session_state:
        st.session_state.image_result = None
    if 'video_result' not in st.session_state:
        st.session_state.video_result = None
    if 'ai_interpretation' not in st.session_state:
        st.session_state.ai_interpretation = None
    
    tab1, tab2, tab3 = st.tabs(["Upload Files", "Capture Live", "Results"])
    
    with tab1:
        
        st.header("Demo Files")
        with st.expander("Use Demo Files Instead of Uploading"):
            demo_files = load_demo_files()
            
            st.subheader("Select Demo Audio")
            demo_audio_selection = st.selectbox(
                "Choose a demo audio file",
                options=list(demo_files["audio"].keys()),
                index=None,
                format_func=lambda x: f"{x} - {demo_files['audio'][x]['description']}" if x else "Select an option"
            )
            
            if demo_audio_selection and st.button("Use Selected Audio"):
                audio_path = demo_files["audio"][demo_audio_selection]["path"]
                if os.path.exists(audio_path):
                    st.session_state.audio_file = audio_path
                    st.audio(audio_path)
                    
                    with st.spinner("Extracting audio features..."):
                        params = get_parameters()
                        st.session_state.features = extract_features(audio_path, params)
                        st.success(f"Using demo audio: {demo_audio_selection}")
                else:
                    st.error(f"Demo file not found: {audio_path}")
            
            st.subheader("Select Demo Image")
            demo_image_selection = st.selectbox(
                "Choose a demo image file",
                options=list(demo_files["image"].keys()),
                index=None,
                format_func=lambda x: f"{x} - {demo_files['image'][x]['description']}" if x else "Select an option"
            )
            
            if demo_image_selection and st.button("Use Selected Image"):
                image_path = demo_files["image"][demo_image_selection]["path"]
                if os.path.exists(image_path):
                    image = Image.open(image_path)
                    st.image(image, caption=f"Demo Image: {demo_image_selection}", use_container_width=True)
                    
                    with st.spinner("Analyzing image..."):
                        st.session_state.image_result = analyze_image_with_imagga(image)
                        st.success(f"Using demo image: {demo_image_selection}")
                else:
                    st.error(f"Demo file not found: {image_path}")
            
            st.subheader("Select Demo Video")
            demo_video_selection = st.selectbox(
                "Choose a demo video file",
                options=list(demo_files["video"].keys()),
                index=None,
                format_func=lambda x: f"{x} - {demo_files['video'][x]['description']}" if x else "Select an option"
            )
            
            if demo_video_selection and st.button("Use Selected Video"):
                video_path = demo_files["video"][demo_video_selection]["path"]
                if os.path.exists(video_path):
                    st.video(video_path)
                    
                    with st.spinner("Analyzing video (this may take a moment)..."):
                        with open(video_path, "rb") as f:
                            video_bytes = f.read()
                        video_file = io.BytesIO(video_bytes)
                        video_file.name = demo_video_selection
                        
                        st.session_state.video_result = analyze_video_with_yolo(video_file)
                        st.success(f"Using demo video: {demo_video_selection}")
                else:
                    st.error(f"Demo file not found: {video_path}")

        st.header("Upload Audio, Image, and Video")
        
        uploaded_audio = st.file_uploader("Upload Audio File", type=['wav', 'mp3', 'ogg'])
        if uploaded_audio is not None:
            st.session_state.audio_file = uploaded_audio
            st.audio(uploaded_audio)
            
            with st.spinner("Extracting audio features..."):
                params = get_parameters()
                st.session_state.features = extract_features(uploaded_audio, params)
                
                if st.session_state.features is not None:
                    plt.figure(figsize=(10, 2))
                    plt.plot(st.session_state.features['audio'])
                    plt.title("Audio Waveform Preview")
                    plt.tight_layout()
                    st.pyplot(plt)
                
                st.success("Audio features extracted successfully.")
        
        uploaded_image = st.file_uploader("Upload Image File", type=['jpg', 'jpeg', 'png'])
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
            
            with st.spinner("Analyzing image..."):
                st.session_state.image_result = analyze_image_with_imagga(Image.open(uploaded_image))
                if "error" in st.session_state.image_result:
                    st.error(st.session_state.image_result["error"])
                else:
                    st.success("Image analyzed successfully.")
        
        uploaded_video = st.file_uploader("Upload Video File", type=['mp4', 'mov', 'avi'])
        if uploaded_video is not None:
            st.video(uploaded_video)
            
            with st.spinner("Analyzing video (this may take a moment)..."):
                st.session_state.video_result = analyze_video_with_yolo(uploaded_video)
                if "error" in st.session_state.video_result:
                    st.error(st.session_state.video_result["error"])
                else:
                    st.success("Video analyzed successfully.")
        
        if st.button("Process Uploaded Files"):
            if st.session_state.features is not None:
                with st.spinner("Processing..."):
                    metadata = st.session_state.metadata
                    selected_categories = []  
                    selected_subcategories = [] 
                    
                    st.session_state.audio_result = predict_with_filters(
                        st.session_state.features,
                        st.session_state.current_model,
                        metadata,
                        selected_categories,
                        selected_subcategories
                    )
                    
                    if (st.session_state.audio_result is not None and 
                        (st.session_state.image_result is not None or st.session_state.video_result is not None)):
                        
                        image_result = st.session_state.image_result if st.session_state.image_result is not None else {}
                        video_result = st.session_state.video_result if st.session_state.video_result is not None else {}
                        
                        st.session_state.ai_interpretation = get_ai_interpretation(
                            st.session_state.audio_result,
                            image_result,
                            video_result
                        )
                
                st.success("Processing complete! Please go to the Results tab.")
            else:
                st.warning("Please upload an audio file first.")
    
    with tab2:
        st.header("Capture Live")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Record Audio")
            if st.button("Record 5 Seconds of Audio"):

                with st.spinner("Please Wait..."):
                    time.sleep(1) 
                    st.session_state.audio_file = "simulated_audio.wav" 
                    st.warning("The streamlit does not support audio recording function - Use File Upload Feature !")
        
        with col2:
            st.subheader("Capture Image/Video")
            capture_choice = st.radio("Capture type:", ["Image", "Video (5 seconds)"])
            
            if st.button(f"Capture {capture_choice}"):

                with st.spinner(f"Please Wait..."):
                    if capture_choice == "Video (5 seconds)":
                        time.sleep(1) 
                    time.sleep(1) 
                    st.warning(f"The streamlit does not support live image capturing function - Use File Upload Feature !")
        
        if st.button("Process Live Capture"):
            st.warning("Live capture processing is a placeholder in this demo. Please use the Upload Files tab for actual processing.")
    
    with tab3:
        st.header("Analysis Results")
        
        if (st.session_state.audio_result is not None and 
            (st.session_state.image_result is not None or st.session_state.video_result is not None) and
            st.session_state.ai_interpretation is not None):
            
            st.subheader("Verification Summary")
            summary_fig = create_verification_summary_visualization(
                st.session_state.audio_result,
                st.session_state.image_result if st.session_state.image_result is not None else {},
                st.session_state.video_result if st.session_state.video_result is not None else {},
                st.session_state.ai_interpretation
            )
            st.plotly_chart(summary_fig, use_container_width=True, key="summary_chart")
            
            st.info(st.session_state.ai_interpretation)
        
            if (st.session_state.audio_result is not None or 
                st.session_state.image_result is not None or 
                st.session_state.video_result is not None):
                
                detail_tabs = st.tabs(["Audio Analysis", "Image Analysis", "Video Analysis"])
                
                with detail_tabs[0]:
                    if st.session_state.audio_result is not None:
                        st.subheader("Audio Classification")
                        
                        audio_class_fig = visualize_audio_classifications(st.session_state.audio_result)
                        st.plotly_chart(audio_class_fig, use_container_width=True, key="audio_class_chart")
                        
                        if st.checkbox("Show detailed audio features"):
                            st.subheader("Audio Feature Visualization")
                            feature_fig = visualize_audio_features(st.session_state.audio_result['features'])
                            st.plotly_chart(feature_fig, use_container_width=True, key="audio_feature_chart")
                        
                        if st.checkbox("Show raw classification data"):
                            audio_result = st.session_state.audio_result
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Category:** {audio_result['category']} ({audio_result['category_confidence']*100:.2f}%)")
                                st.write("**Top Categories:**")
                                for cat, conf in audio_result['top_categories']:
                                    st.write(f"- {cat}: {conf*100:.2f}%")
                            with col2:
                                st.write(f"**Subcategory:** {audio_result['subcategory']} ({audio_result['subcategory_confidence']*100:.2f}%)")
                                st.write("**Top Subcategories:**")
                                for subcat, conf in audio_result['top_subcategories']:
                                    st.write(f"- {subcat}: {conf*100:.2f}%")
                    else:
                        st.info("No audio analysis results available. Please upload and process an audio file.")
                
                with detail_tabs[1]:
                    if st.session_state.image_result is not None and "objects" in st.session_state.image_result:
                        st.subheader("Image Analysis (Imagga)")
                        
                        if 'image_data' in st.session_state:
                            st.image(st.session_state.image_data, caption="Analyzed Image", use_container_width=True)
                        
                        image_objects = st.session_state.image_result["objects"]
                        if image_objects:
                            object_fig = visualize_object_detection(image_objects, "Detected Objects in Image")
                            st.plotly_chart(object_fig, use_container_width=True, key="image_object_chart")
                        else:
                            st.write("No objects detected in the image.")
                    else:
                        st.info("No image analysis results available. Please upload and process an image file.")
                
                with detail_tabs[2]:
                    if st.session_state.video_result is not None and "objects" in st.session_state.video_result:
                        st.subheader("Video Analysis (YOLO)")
                        
                        video_objects = st.session_state.video_result["objects"]
                        if video_objects:
                            video_fig = visualize_object_detection(video_objects, "Detected Objects in Video")
                            st.plotly_chart(video_fig, use_container_width=True, key="video_object_chart")
                            
                            if "frames_with_detections" in st.session_state.video_result and st.session_state.video_result["frames_with_detections"]:
                                st.subheader("Object Detection in Video Frames")
                                frames_fig = visualize_frames_with_detections(st.session_state.video_result["frames_with_detections"])
                                if frames_fig:
                                    st.pyplot(frames_fig)
                        else:
                            st.write("No objects detected in the video.")
                    else:
                        st.info("No video analysis results available. Please upload and process a video file.")
                
                # Cross-Modal Comparison section
                st.subheader("Cross-Modal Comparison")
                
                comparison_data = []
                
                if st.session_state.audio_result is not None:
                    audio_cat = st.session_state.audio_result['category']
                    audio_subcat = st.session_state.audio_result['subcategory']
                    comparison_data.append({
                        "Source": "Audio",
                        "Category": audio_cat,
                        "Subcategory/Object": audio_subcat,
                        "Confidence": st.session_state.audio_result['subcategory_confidence'] * 100
                    })
                
                if st.session_state.image_result is not None and "objects" in st.session_state.image_result:
                    for i, obj in enumerate(st.session_state.image_result["objects"][:3]):  # Top 3 objects
                        comparison_data.append({
                            "Source": f"Image Object {i+1}",
                            "Category": "Visual Object",
                            "Subcategory/Object": obj["tag"],
                            "Confidence": obj["confidence"]
                        })
                
                if st.session_state.video_result is not None and "objects" in st.session_state.video_result:
                    for i, obj in enumerate(st.session_state.video_result["objects"][:3]):  # Top 3 objects
                        comparison_data.append({
                            "Source": f"Video Object {i+1}",
                            "Category": "Visual Object",
                            "Subcategory/Object": obj["tag"],
                            "Confidence": obj["confidence"] * 100
                        })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    fig = px.bar(comparison_df, 
                            x="Source", 
                            y="Confidence", 
                            color="Subcategory/Object",
                            title="Cross-Modal Comparison",
                            labels={"Confidence": "Confidence (%)", "Source": "Analysis Source"})
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key="comparison_chart")
                
                    st.write("Comparison Summary Table:")
                    st.dataframe(comparison_df)
                
                # Download report section
                report = create_download_report()
                if report:
                    st.download_button(
                        label="Download Analysis Report",
                        data=report,
                        file_name=f"audio_visual_analysis_{time.strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
        else:
            st.info("No analysis results available yet. Please upload and process files in the Upload Files tab.")

    st.markdown("""
        <div style="text-align: center; margin-top: 50px; padding: 20px; font-size: 0.8em; color: #94a3b8;">
            <p>Audio-Visual Verification System - An advanced media analysis tool</p>
        </div>
        """, unsafe_allow_html=True)
    
def create_download_report():
    if (st.session_state.audio_result is not None or 
        st.session_state.image_result is not None or 
        st.session_state.video_result is not None):
        
        report = io.StringIO()
        
        report.write("# Audio-Visual Verification System - Analysis Report\n\n")
        report.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if st.session_state.ai_interpretation:
            report.write("## AI Interpretation\n\n")
            report.write(f"{st.session_state.ai_interpretation}\n\n")
        
        if st.session_state.audio_result:
            report.write("## Audio Analysis Results\n\n")
            audio_result = st.session_state.audio_result
            report.write(f"Category: {audio_result['category']} ({audio_result['category_confidence']*100:.2f}%)\n")
            report.write(f"Subcategory: {audio_result['subcategory']} ({audio_result['subcategory_confidence']*100:.2f}%)\n\n")
            
            report.write("Top Categories:\n")
            for cat, conf in audio_result['top_categories']:
                report.write(f"- {cat}: {conf*100:.2f}%\n")
            
            report.write("\nTop Subcategories:\n")
            for subcat, conf in audio_result['top_subcategories']:
                report.write(f"- {subcat}: {conf*100:.2f}%\n")
            
            report.write("\n")
        
        if st.session_state.image_result and "objects" in st.session_state.image_result:
            report.write("## Image Analysis Results\n\n")
            image_objects = st.session_state.image_result["objects"]
            
            if image_objects:
                report.write("Detected Objects:\n")
                for obj in image_objects:
                    report.write(f"- {obj['tag']}: {obj['confidence']:.2f}%\n")
            else:
                report.write("No objects detected in the image.\n")
            
            report.write("\n")
        
        if st.session_state.video_result and "objects" in st.session_state.video_result:
            report.write("## Video Analysis Results\n\n")
            video_objects = st.session_state.video_result["objects"]
            
            if video_objects:
                report.write("Detected Objects:\n")
                for obj in video_objects:
                    report.write(f"- {obj['tag']}: {obj['confidence']*100:.2f}%\n")
            else:
                report.write("No objects detected in the video.\n")
            
            report.write("\n")
        
        return report.getvalue()
    
    return None

    

if __name__ == "__main__":

    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import io
    import time
    
    main()