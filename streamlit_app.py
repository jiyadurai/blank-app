import streamlit as st
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

df = pd.read_csv("post_ig.csv")
def get_common_name(predicted_class_index, df, animal_name_list):
    scientific_name = animal_name_list[predicted_class_index]
    match = df[df['scientific_name'] == scientific_name]
    if not match.empty:
        return match['common_name'].iloc[0]
    else:
        return "Unknown"
animal_name_list = df['scientific_name'].unique().tolist()


def normalize(x, minimum, maximum):
    return (x-minimum)/(maximum-minimum)

def time_of_day_encoder(part_of_day):
    cats = ['morning', 'afternoon', 'evening', 'night']
    ind = cats.index(part_of_day)
    return torch.nn.functional.one_hot(torch.tensor(ind), num_classes=len(cats)).float()

class MyAnimalClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MyAnimalClassifier, self).__init__()

        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.resnet_out_dim = resnet.fc.in_features

        self.fc = nn.Sequential(
            # add in the features for latitude, longitude, part of day
            # slowly decrease output until we get to scientific name
            nn.Linear(in_features=self.resnet_out_dim + 2 + 4, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            # takes in 256, out 128
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            # Final step; output is the number of species
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, image, location, time_of_day):
        x_img = self.resnet_backbone(image)
        x_img = x_img.view(x_img.size(0), -1)
        time_of_day = time_of_day.unsqueeze(0)
        # st.write(
        #     f"Shape of x_img: {x_img.shape}, Shape of location: {location.shape}, Shape of time_of_day: {time_of_day.shape}",
        # )
        # print()
        # print("Shape of location:", location.shape)
        # print("Shape of time_of_day:", time_of_day.shape)

        x = torch.cat((x_img, location, time_of_day), dim=1)
        return self.fc(x)
    

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = torch.load('Models/full_resnet_trained.pth', weights_only=False, map_location=torch.device('cpu'))
# st.write(type(model))
model.eval()

st.title("Florida Mammalian Classification")
st.write(
    "This is a tool that can classify the mammals of Florida, based on user submitted images."
)

options = ["Morning", "Afternoon", "Evening", "Night"]
selected_option = st.selectbox("Select an option:", options)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
image_tensor = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    st.image(image, caption="Uploaded Image.")
    
florida_image = Image.open("florida.png")
coordinates = streamlit_image_coordinates(florida_image, key="my_image", width=740, height=630)

if coordinates:
    st.write(f"Clicked coordinates: x={coordinates['x']}, y={coordinates['y']}")

if uploaded_file is not None and coordinates and selected_option:
    with torch.no_grad():
        latitude = normalize(coordinates['y'], 0, 638)
        longitude = normalize(coordinates['x'], 0, 743)
        location = torch.tensor([latitude, longitude]).unsqueeze(0)
        part_of_day = time_of_day_encoder(selected_option.lower())
        output = model(image_tensor, location, part_of_day)
        _, predicted = torch.max(output, 1)

        st.write(f"Predicted species: {get_common_name(predicted, df, animal_name_list)}")


