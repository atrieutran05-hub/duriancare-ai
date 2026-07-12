import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN TRANG WEB
# ==========================================
st.set_page_config(
    page_title="Bác Sĩ Sầu Riêng AI", 
    page_icon="🌳", 
    layout="centered"
)

# Danh sách các loại bệnh (Theo đúng thứ tự Alphabet lúc máy học trong Colab)
CLASS_NAMES = [
    "ALGAL_LEAF_SPOT", 
    "ALLOCARIDARA_ATTACK", 
    "HEALTHY_LEAF", 
    "LEAF_BLIGHT", 
    "PHOMOPSIS_LEAF_SPOT"
]

# ==========================================
# 2. HÀM NẠP DỮ LIỆU (Có dùng Cache để web chạy mượt)
# ==========================================
@st.cache_resource
def load_model():
    """Tải mô hình AI đã huấn luyện (.keras)"""
    return tf.keras.models.load_model('durian_care_model.keras')

@st.cache_data
def load_database():
    """Tải từ điển bệnh (.json)"""
    with open('database.json', 'r', encoding='utf-8') as f:
        return json.load(f)

# Bắt đầu nạp model và database ẩn dưới nền
try:
    model = load_model()
    database = load_database()
except Exception as e:
    st.error(f"❌ Lỗi khởi tạo hệ thống! Ní kiểm tra lại xem đã up đủ file model và json lên GitHub chưa nha.\nLỗi chi tiết: {e}")
    st.stop() # Dừng app nếu không có file

# ==========================================
# 3. XÂY DỰNG GIAO DIỆN CHÍNH
# ==========================================
st.title("🌳 Trợ lý AI: Bác Sĩ Sầu Riêng")
st.markdown("""
Chào mừng đến với hệ thống nhận diện bệnh trên lá sầu riêng! 
Chỉ cần tải lên một bức ảnh chụp lá sầu riêng, AI sẽ tự động "bắt mạch" và đưa ra cách trị bệnh chuẩn xác.
""")

# Khu vực upload ảnh
uploaded_file = st.file_uploader("Tải ảnh lá sầu riêng lên đây (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 3.1 Hiển thị ảnh người dùng tải lên
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Ảnh lá sầu riêng đang phân tích...", use_container_width=True)

    # Nút bấm kích hoạt AI
    if st.button("🔍 Chuẩn đoán ngay", use_container_width=True):
        with st.spinner('Bác sĩ đang "khám" lá... Chờ một xíu nha!'):
            try:
                # 3.2 Tiền xử lý ảnh (Đưa về kích thước 224x224 chuẩn của model)
                img_resized = image.resize((224, 224))
                img_array = tf.keras.utils.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) # Thêm chiều batch (1, 224, 224, 3)

                # 3.3 AI bắt đầu phán đoán
                predictions = model.predict(img_array)
                confidence_scores = predictions[0] # Tỉ lệ % của từng bệnh
                
                # Tìm ra bệnh có % cao nhất
                predicted_class_index = np.argmax(confidence_scores)
                predicted_class_name = CLASS_NAMES[predicted_class_index]
                confidence = int(confidence_scores[predicted_class_index] * 100) # Đổi ra số tròn 100%

                # 3.4 Lấy thông tin tiếng Việt từ file json
                disease_info = database.get(predicted_class_name, {})
                ten_benh_vi = disease_info.get("name_vi", "Không xác định")
                cach_tri = disease_info.get("treatment", "Chưa có thông tin cách trị.")

                # 3.5 Hiển thị kết quả ra màn hình
                st.divider()
                st.subheader("📋 Kết quả chuẩn đoán:")
                
                if predicted_class_name == "HEALTHY_LEAF":
                    # Nếu lá khỏe mạnh -> Báo màu xanh
                    st.success(f"🎉 **Tình trạng:** {ten_benh_vi} (Độ tự tin: {confidence}%)")
                    st.info(f"💡 **Lời khuyên:** {cach_tri}")
                else:
                    # Nếu có bệnh -> Báo màu đỏ cảnh báo
                    st.error(f"⚠️ **Tình trạng:** {ten_benh_vi} (Độ tự tin: {confidence}%)")
                    st.warning(f"💊 **Cách xử lý:** {cach_tri}")

            except Exception as e:
                st.error(f"Đã có lỗi xảy ra trong lúc tính toán: {e}")
