# Audio-Visual Verification System

## Overview
The Audio-Visual Verification System is an advanced media analysis tool that verifies whether audio content matches visual content in images and videos. Using machine learning and AI, it detects mismatches between audio and visual elements, helping to identify potentially misleading or manipulated media.

## Features
- **Audio Analysis**: Classifies audio into categories and subcategories using MFCC and mel spectrogram features
- **Image Analysis**: Detects objects in images using the Imagga API with confidence scores
- **Video Analysis**: Identifies objects in videos using YOLOv8 object detection with frame-by-frame tracking
- **AI Interpretation**: Compares audio and visual data to determine if they match
- **Built-in Demo Files**: Test the system with pre-loaded audio, image, and video examples
- **Interactive Visualizations**: View detailed analysis with interactive charts and visualizations

## Use Cases
- Content verification for media platforms
- Detecting manipulated or deepfake content
- Educational tool for media literacy
- Research on audio-visual correspondence
- Forensic analysis of multimedia content

## Installation

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/audio-visual-verification.git
   cd audio-visual-verification
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up API keys:
   - Register for an [Imagga API key](https://imagga.com/auth/signup)
   - (Optional) Get an [OpenAI API key](https://platform.openai.com/signup) for enhanced AI interpretation
   - Create a `.env` file in the project root with your API keys:
     ```
     IMAGGA_API_KEY=your_imagga_key
     IMAGGA_API_SECRET=your_imagga_secret
     OPENAI_API_KEY=your_openai_key
     GROQ_API_KEY=your_groq_key
     ```

5. Download YOLOv8 model (this will happen automatically on first run, but you can pre-download):
   ```bash
   pip install ultralytics
   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
   ```

6. Create demo files directory structure:
   ```bash
   mkdir -p demo_files/audio demo_files/images demo_files/videos
   ```

7. Add your demo files to the respective directories.

## Running the Application

Launch the Streamlit app:
```bash
streamlit run proofing.py
```

The application will be available at http://localhost:8501 in your web browser.

## How to Use

1. **Using Demo Files**:
   - Expand the "Use Demo Files" section
   - Select from available demo audio, images, and videos
   - Click the corresponding "Use Selected" button
   - Process the files with the "Process Uploaded Files" button

2. **Using Your Own Files**:
   - Upload audio (.wav, .mp3, .ogg), image (.jpg, .png), and video (.mp4, .mov, .avi) files
   - Files will be processed individually
   - Click "Process Uploaded Files" to analyze all uploaded content

3. **View the Results**:
   - Navigate to the "Results" tab
   - Check the verification summary with AI interpretation
   - Explore detailed audio, image, and video analysis
   - Download a comprehensive report of the findings

## Technology Stack
- Streamlit (UI framework)
- TensorFlow (audio classification)
- YOLOv8 (object detection in videos)
- Librosa (audio processing)
- Plotly and Matplotlib (data visualization)
- OpenCV (image and video processing)
- Imagga API (image classification)
- LLM APIs (OpenAI/Groq for interpretation)
- Pandas and NumPy (data handling)

## Requirements

See the complete list in `requirements.txt`:
```
streamlit>=1.22.0
numpy>=1.22.0
opencv-python>=4.5.5
tensorflow>=2.9.0
ultralytics>=0.3.0
librosa>=0.9.2
pillow>=9.1.0
matplotlib>=3.5.1
seaborn>=0.12.0
plotly>=5.6.0
pandas>=1.4.2
requests>=2.27.1
python-dotenv>=0.20.0
openai>=1.3.0
```

## License
MIT License

## Contributors
- [Lokesh B](https://github.com/LokeshBhaskarNR)

## Acknowledgements
- YOLOv8 by Ultralytics
- Imagga API for image recognition
- OpenAI and Groq for LLM capabilities
