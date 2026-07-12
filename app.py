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
    .main-title {
        font-size: 42px;
        color: #2e7d32;
        text-align: center;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        color: #888888;
        font-size: 16px;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

CLASS_NAMES = [
    "ALGAL_LEAF_SPOT", "ALLOCARIDARA_ATTACK", 
    "HEALTHY_LEAF", "LEAF_BLIGHT", "PHOMOPSIS_LEAF_SPOT"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('durian_care_model.keras')

@st.cache_data
def load_database():
    with open('database.json', 'r', encoding='utf-8') as f:
        return json.load(f)

try:
    model = load_model()
    database = load_database()
except Exception as e:
    st.error(f"❌ Lỗi khởi tạo hệ thống! Vui lòng kiểm tra file model và json. Lỗi: {e}")
    st.stop()

st.markdown('<div class="main-title">🌳 Bác Sĩ Sầu Riêng</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Chuyên gia AI chẩn đoán bệnh qua hình ảnh, nhanh chóng & chính xác.</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📸 Chụp ảnh trực tiếp", "📂 Tải ảnh từ thư viện"])

image_to_process = None

with tab1:
    st.info("💡 Mẹo: Đưa điện thoại lại gần lá sầu riêng, lấy nét rõ để AI nhìn chuẩn nhất!")
    camera_image = st.camera_input("Nhấn vào đây để mở Camera")
    if camera_image:
        image_to_process = camera_image

with tab2:
    uploaded_file = st.file_uploader("Chọn ảnh lá sầu riêng (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_to_process = uploaded_file

if image_to_process is not None:
    image = Image.open(image_to_process).convert('RGB')
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="📸 Ảnh đang được phân tích...", use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    with st.spinner('AI đang quét tế bào lá... Chờ một xíu nha!'):
        try:
            img_resized = image.resize((224, 224))
            img_array = tf.keras.utils.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            confidence_scores = predictions[0]
            
            predicted_class_index = np.argmax(confidence_scores)
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = int(confidence_scores[predicted_class_index] * 100)

            disease_info = database.get(predicted_class_name, {})
            ten_benh_vi = disease_info.get("name_vi", "Không xác định")
            cach_tri = disease_info.get("treatment", "Chưa có thông tin cách trị.")

            st.divider()
            st.markdown("<h3 style='text-align: center; color: #ff9800;'>📋 KẾT QUẢ CHUẨN ĐOÁN</h3>", unsafe_allow_html=True)
            
            if predicted_class_name == "HEALTHY_LEAF":
                st.success(f"🎉 **Tình trạng:** {ten_benh_vi} (Khả năng: {confidence}%)")
                st.info(f"💡 **Lời khuyên:** {cach_tri}")
                st.balloons()
            else:
                st.error(f"⚠️ **Phát hiện bệnh:** {ten_benh_vi} (Khả năng: {confidence}%)")
                st.warning(f"💊 **Phác đồ điều trị:** {cach_tri}")

        except Exception as e:
            st.error(f"Lỗi tính toán: {e}")
