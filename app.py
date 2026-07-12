import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN
# ==========================================
st.set_page_config(
    page_title="Bác Sĩ Sầu Riêng AI", 
    page_icon="🌳", 
    layout="centered"
)

# CSS làm đẹp cho mobile
st.markdown("""
    <style>
    .main-title { color: #2e7d32; text-align: center; font-weight: 800; font-size: 30px; }
    .stTabs [data-baseweb="tab-list"] { justify-content: center; }
    </style>
""", unsafe_allow_html=True)

# Danh sách bệnh (Phải khớp 100% với lúc train)
CLASS_NAMES = [
    "ALGAL_LEAF_SPOT", "ALLOCARIDARA_ATTACK", 
    "HEALTHY_LEAF", "LEAF_BLIGHT", "PHOMOPSIS_LEAF_SPOT"
]

# ==========================================
# 2. HÀM TỐI ƯU (Tiết kiệm RAM)
# ==========================================
@st.cache_resource
def load_model():
    # compile=False giúp app không tốn RAM để load các thông số training không cần thiết
    return tf.keras.models.load_model('durian_care_model.keras', compile=False)

@st.cache_data
def load_database():
    with open('database.json', 'r', encoding='utf-8') as f:
        return json.load(f)

# Load resources
try:
    model = load_model()
    database = load_database()
except Exception as e:
    st.error(f"❌ Lỗi: {e}")
    st.stop()

# ==========================================
# 3. GIAO DIỆN & XỬ LÝ
# ==========================================
st.markdown('<div class="main-title">🌳 Bác Sĩ Sầu Riêng</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📸 Camera", "📂 Thư viện"])

image_to_process = None

with tab1:
    camera_image = st.camera_input("Chụp ảnh lá")
    if camera_image:
        image_to_process = camera_image

with tab2:
    uploaded_file = st.file_uploader("Chọn ảnh từ máy", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_to_process = uploaded_file

# Xử lý tự động ngay khi có ảnh
if image_to_process is not None:
    image = Image.open(image_to_process).convert('RGB')
    st.image(image, use_container_width=True)
    
    with st.spinner('AI đang chẩn đoán...'):
        try:
            # Tiền xử lý (đúng chuẩn 224x224)
            img_resized = image.resize((224, 224))
            img_array = tf.keras.utils.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)

            # Dự đoán
            predictions = model.predict(img_array)
            idx = np.argmax(predictions[0])
            confidence = predictions[0][idx] # Lấy độ tin cậy (0.0 đến 1.0)
            label = CLASS_NAMES[idx]
            
            # --- CƠ CHẾ KIỂM TRA ĐỘ TIN CẬY (FIX LỖI SAI BẬY) ---
            # Nếu AI chỉ tự tin dưới 60%, ép nó báo "Không rõ" thay vì đoán mò
            if confidence < 0.6:
                st.warning("⚠️ AI không chắc chắn lắm về bệnh này. Hãy thử chụp lại ở góc khác hoặc nơi có ánh sáng tốt hơn nhé!")
            else:
                info = database.get(label, {"name_vi": "Khác", "treatment": "Liên hệ chuyên gia."})
                conf_pct = int(confidence * 100)
                
                st.divider()
                if label == "HEALTHY_LEAF":
                    st.success(f"✅ **{info['name_vi']}** (Độ tin cậy: {conf_pct}%)")
                    st.balloons()
                else:
                    st.error(f"⚠️ **{info['name_vi']}** (Độ tin cậy: {conf_pct}%)")
                
                st.info(f"💡 **Cách xử lý:** {info['treatment']}")
            
        except Exception as e:
            st.error(f"Lỗi: {e}")
