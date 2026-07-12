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

CLASS_NAMES = [
    "ALGAL_LEAF_SPOT", 
    "ALLOCARIDARA_ATTACK", 
    "HEALTHY_LEAF", 
    "LEAF_BLIGHT", 
    "PHOMOPSIS_LEAF_SPOT"
]

@st.cache_resource
def load_model():
    """Tải mô hình AI đã huấn luyện (.keras)"""
    return tf.keras.models.load_model('durian_care_model.keras')

@st.cache_data
def load_database():
    """Tải từ điển bệnh (.json)"""
    with open('database.json', 'r', encoding='utf-8') as f:
        return json.load(f)

try:
    model = load_model()
    database = load_database()
except Exception as e:
    st.error(f"❌ Lỗi khởi tạo hệ thống! Ní kiểm tra lại xem đã up đủ file model và json lên GitHub chưa nha.\nLỗi chi tiết: {e}")
    st.stop() 
    
st.title("🌳 Trợ lý AI: Bác Sĩ Sầu Riêng")
st.markdown("""
Chào mừng đến với hệ thống nhận diện bệnh trên lá sầu riêng! 
Chỉ cần tải lên một bức ảnh chụp lá sầu riêng, AI sẽ tự động "bắt mạch" và đưa ra cách trị bệnh chuẩn xác.
""")

uploaded_file = st.file_uploader("Tải ảnh lá sầu riêng lên đây (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Ảnh lá sầu riêng đang phân tích...", use_container_width=True)

    if st.button("🔍 Chuẩn đoán ngay", use_container_width=True):
        with st.spinner('Bác sĩ đang "khám" lá... Chờ một xíu nha!'):
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
                st.subheader("📋 Kết quả chuẩn đoán:")
                
                if predicted_class_name == "HEALTHY_LEAF":
                    st.success(f"🎉 **Tình trạng:** {ten_benh_vi} (Độ tự tin: {confidence}%)")
                    st.info(f"💡 **Lời khuyên:** {cach_tri}")
                else:
                    st.error(f"⚠️ **Tình trạng:** {ten_benh_vi} (Độ tự tin: {confidence}%)")
                    st.warning(f"💊 **Cách xử lý:** {cach_tri}")

            except Exception as e:
                st.error(f"Đã có lỗi xảy ra trong lúc tính toán: {e}")
