import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

st.set_page_config(
    page_title="Bác Sĩ Sầu Riêng AI", 
    page_icon="🌳", 
    layout="centered"
)

st.markdown("""
    <style>
    .main-title { color: #2e7d32; text-align: center; font-weight: 800; font-size: 35px; }
    .sub-title { text-align: center; color: #666; margin-bottom: 20px; }
    .result-box { padding: 20px; border-radius: 15px; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

CLASS_NAMES = [
    "ALGAL_LEAF_SPOT", "ALLOCARIDARA_ATTACK", 
    "HEALTHY_LEAF", "LEAF_BLIGHT", "PHOMOPSIS_LEAF_SPOT"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('durian_care_model.keras', compile=False)

@st.cache_data
def load_database():
    with open('database.json', 'r', encoding='utf-8') as f:
        return json.load(f)

try:
    model = load_model()
    database = load_database()
except Exception as e:
    st.error(f"Lỗi hệ thống: {e}")
    st.stop()

st.markdown('<div class="main-title">🌳 Bác Sĩ Sầu Riêng</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI chẩn đoán bệnh tức thì</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📸 Camera", "📂 Thư viện"])

image_to_process = None

with tab1:
    camera_image = st.camera_input("Chụp ảnh lá sầu riêng")
    if camera_image:
        image_to_process = camera_image

with tab2:
    uploaded_file = st.file_uploader("Hoặc chọn ảnh từ máy", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_to_process = uploaded_file

if image_to_process is not None:
    image = Image.open(image_to_process).convert('RGB')
    st.image(image, use_container_width=True)
    
    with st.spinner('Đang phân tích...'):
        try:
            img_resized = image.resize((224, 224))
            img_array = tf.keras.utils.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            idx = np.argmax(predictions[0])
            label = CLASS_NAMES[idx]
            conf = int(predictions[0][idx] * 100)

            info = database.get(label, {"name_vi": "Khác", "treatment": "Liên hệ chuyên gia."})
            
            st.divider()
            if label == "HEALTHY_LEAF":
                st.success(f"✅ **{info['name_vi']}** ({conf}%)")
                st.balloons()
            else:
                st.error(f"⚠️ **{info['name_vi']}** ({conf}%)")
            
            st.info(f"💡 **Cách xử lý:** {info['treatment']}")
            
        except Exception as e:
            st.error(f"Lỗi: {e}")
