import streamlit as st
import numpy as np
import cv2
import pickle
from PIL import Image

# Set page config at the very beginning
st.set_page_config(page_title="Traffic Sign Classifier", page_icon="🚦", layout="wide")

# Load the pickled model
@st.cache_resource
def load_model():
    with open("traffic_sign_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Class names
class_names = {
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 
    3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)', 
    9:'No passing', 10:'No passing for vehicles over 3.5 metric tons', 
    11:'Right-of-way at the next intersection', 12:'Priority road', 13:'Yield', 
    14:'Stop', 15:'No vehicles', 16:'Vehicles over 3.5 metric tons prohibited', 
    17:'No entry', 18:'General caution', 19:'Dangerous curve to the left', 
    20:'Dangerous curve to the right', 21:'Double curve', 22:'Bumpy road', 
    23:'Slippery road', 24:'Road narrows on the right', 25:'Road work', 
    26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing', 
    29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing', 
    32:'End of all speed and passing limits', 33:'Turn right ahead', 
    34:'Turn left ahead', 35:'Ahead only', 36:'Go straight or right', 
    37:'Go straight or left', 38:'Keep right', 39:'Keep left', 
    40:'Roundabout mandatory', 41:'End of no passing', 
    42:'End of no passing by vehicles over 3.5 metric tons'
}

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

st.title("🚦 Traffic Sign Classifier")

st.write("""
Upload an image of a traffic sign, and our AI model will classify it!
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (32, 32))
    img_preprocessed = preprocess_image(img_resized)
    img_preprocessed = img_preprocessed.reshape(1, 32, 32, 1)
    
    # Make prediction
    prediction = model.predict(img_preprocessed)
    class_index = np.argmax(prediction)
    class_name = class_names[class_index]
    probability = np.max(prediction) * 100
    
    # Display results
    st.write("## Prediction Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### Class ID: {class_index}")
        st.markdown(f"### Class Name: {class_name}")
    
    with col2:
        st.markdown(f"### Probability: {probability:.2f}%")
        # Create a progress bar for probability
        st.progress(probability / 100)
    
    # Display top 5 predictions
    st.write("## Top 5 Predictions")
    top5_indices = np.argsort(prediction[0])[-5:][::-1]
    for i, idx in enumerate(top5_indices):
        st.write(f"{i+1}. {class_names[idx]}: {prediction[0][idx]*100:.2f}%")

st.write("""
### How to use:
1. Upload an image of a traffic sign using the file uploader above.
2. The AI model will process the image and provide its prediction.
3. The top prediction, along with its probability, will be displayed.
4. You'll also see the top 5 predictions for reference.

Note: For best results, use clear images of traffic signs with good lighting and minimal background clutter.
""")

st.write("---")
st.write("Developed with ❤️ using Streamlit and TensorFlow")