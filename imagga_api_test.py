import streamlit as st
import requests
from io import BytesIO
import base64

API_KEY = 'acc_aeaca4a5cd61e4d'
API_SECRET = '0890f560ffe4025210fe54cc56f6a63c'

def get_image_tags(image_bytes):

    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    response = requests.post(
        'https://api.imagga.com/v2/tags',
        auth=(API_KEY, API_SECRET),
        files={'image': ('image.jpg', BytesIO(image_bytes))}
    )
    return response.json()

st.title("Image Tagging with Imagga API")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    image_bytes = uploaded_file.getvalue()
    
    with st.spinner('Analyzing the image...'):
        result = get_image_tags(image_bytes)
    
    if 'result' in result:
        tags = result['result']['tags']
        st.subheader("Identified Tags:")
        for tag in tags:
            st.write(f"{tag['tag']['en']} ({tag['confidence']:.2f}%)")
    else:
        st.error("Error in fetching tags. Please check your API credentials and try again.")
