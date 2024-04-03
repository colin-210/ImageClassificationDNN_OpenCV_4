import streamlit as st
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import requests
from requests.models import MissingSchema
from io import BytesIO


@st.cache_resource
# Function to load model and extract class names from ImageNet dataset
def load_model():
    with open('classification_classes_ILSVRC2012.txt', 'r') as f:
        image_net_names = f.read().split('\n')

    class_names = [name.split(',')[0] for name in image_net_names]

    model = cv2.dnn.readNet(model='DenseNet_121.caffemodel',
                            config='DenseNet_121.prototxt',
                            framework='Caffe')
    return model, class_names


# Function to inference model and return class name with highest confidence
def classify(model, image, class_names):
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    blob = cv2.dnn.blobFromImage(image=image, scalefactor=0.017, size=(224, 224), mean=(104, 117, 123))

    model.setInput(blob)
    outputs = model.forward()
    final_outputs = outputs[0]

    final_outputs = final_outputs.reshape(1000, 1)

    label_id = np.argmax(final_outputs)

    probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))

    final_prob = np.max(probs) * 100

    out_name = class_names[label_id]
    out_text = f"Class: {out_name}, Confidence: {final_prob:.1f}%"

    return out_text


# Header
def header(text):
    st.markdown('<p style="background-color:#0066cc;color:#33ff33;font-size:24px;'
                f'border-radius:2%;" align="center">{text}</p>', unsafe_allow_html=True)

st.title(':red[Image Classification with OpenCV DNN & ImageNet]')
net, class_names = load_model()

img_file_buffer = st.file_uploader("Choose a file of Camera", type=['jpg', 'jpeg', 'png'])
st.text('OR')
url = st.text_input('Enter URL')

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
    st.image(image)

    detections = classify(net, image, class_names)
    header(detections)

elif url != '':
    try:
        response = requests.get(url)
        image = np.array(Image.open(BytesIO(response.content)))
        st.image(image)

        detections = classify(net, image, class_names)
        header(detections)
    except MissingSchema as err:
        st.header('Invalid URL Try Again')
        print(err)
    except UnidentifiedImageError as err:
        st.header('URL has no Image Try Again')
        print(err)

st.subheader(':blue[Try using your phone and taking a picture with your camera!]', divider='rainbow')