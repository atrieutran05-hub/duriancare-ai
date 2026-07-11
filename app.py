import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

st.set_page_config(page_title="DurianCare AI", page_icon="🌳")
st.title("🌳 DurianCare AI - Bác Sĩ Sầu Riêng")

class_names = [
    "Algal leaf spot", 
    "Allocaridara attack", 
    "Healthy leaf", 
    "Leaf blight", 
    "Phomopsis leaf spot"
]

@st.cache_resource
def load_my_model():
    try:
        model = tf.keras.models.load_model('durian_care_model.h5', compile=False)
        return model
    except Exception as e:
        st.error(f"Lỗi nạp model: {e}")
        return None

@st.cache_data
def load_db():
    if os.path.exists('database.json'):
        with open('database.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

model = load_my_model()
db = load_db()

uploaded_file = st.file_uploader("Tải lên ảnh lá sầu riêng:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh kiểm tra', use_container_width=True)
    
    if st.button("Phân tích"):
        if model is None:
            st.error("Model chưa sẵn sàng!")
        else:
            img = image.resize((224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, 0) / 255.0
            
            predictions = model.predict(img_array)
            idx = np.argmax(predictions)
            label = class_names[idx]
            conf = np.max(predictions) * 100
            
            st.write(f"### Kết quả: **{label}** ({conf:.2f}%)")
            
            info = db.get(label, {"name_vi": "Chưa có thông tin", "treatment": "Đang cập nhật"})
            st.write(f"**Tên tiếng Việt:** {info['name_vi']}")
            st.info(f"**Hướng dẫn:** {info['treatment']}")
