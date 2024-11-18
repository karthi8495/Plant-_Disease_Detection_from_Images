import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import os

class CNN_Classification(nn.Module):
    def __init__(self):
        super(CNN_Classification,self).__init__()

        # Convolutional layers(# first convolutional layers)
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1) 
        # first pooling layers
        self.pool = nn.MaxPool2d(
            kernel_size=2,stride=2,padding=0
        )
        # second convolutional layers
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.fc1 = nn.Linear(64*37*37,128)
        self.fc2 = nn.Linear(128,38)

    def  forward(self,x):
         x = self.pool(F.relu(self.conv1(x)))
         x = self.pool(F.relu(self.conv2(x)))
         x = torch.flatten(x,1)
         x = torch.relu(self.fc1(x))
         x = self.fc2(x)
         return x

# Load the trained Pytorch model using st.cache_resource
@st.cache_resource
def load_model(model_path):
    model = CNN_Classification()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


# Define class names based on your trained model (8 classes)
CLASS_NAMES = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']


# Define image transformations

# Define image transformations
def transform_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize((150, 150)),  # Resize to match training
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # Normalize
        ]
    )
    return transform(image).unsqueeze(0)  # Add batch dimension


# Predict function
def predict(image, model):
    image = transform_image(image)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, 1)[0][predicted].item() * 100
        return CLASS_NAMES[predicted.item()], confidence


# Streamlit App

def main():
    st.title(":orange[Plant Disease Detection Application]")
    st.write("Upload an image of a plant leaf to detect the disease type.")

    # Image Upload
    uploaded_file = st.file_uploader("Choose an image...",type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image,caption="Uploaded Image",use_column_width=True)

           # Load Model
            model_path = "D:\GUVI AI & ML\Capstone_Final_Project2_Plant Disease Detection from Images\cnn_model.pth"   # this path should be correct
            num_classes = len(CLASS_NAMES)
            model = load_model(model_path)

            # Prediction button
            if st.button("Predict Disease"):
                 with st.spinner("Processing..."):
                     predicted_class,confidence = predict(image,model)
                     st.success(f"The leaf is predicted to have: **{predicted_class}**with a confidence of **{confidence:.2f}%**.")

        except Exception as e: 
            st.error(f"An error occured: {e}")


if __name__ == "__main__":
    main()
