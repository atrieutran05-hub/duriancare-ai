import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# 1. Cấu hình giao diện App
st.set_page_config(page_title="DurianCare AI", page_icon="🌳", layout="centered")

st.title("🌳 DurianCare AI - Bác Sĩ Sầu Riêng")
st.write("Dự án chẩn đoán bệnh sầu riêng bằng AI qua hình ảnh lá cây.")
st.write("---")

# 2. Nạp model sầu riêng (Dùng cache để App chạy mượt, không nạp lại file nặng khi bấm nút)
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('durian_care_model.keras')

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Lỗi nạp mô hình AI: {e}")

# 3. Danh sách bệnh chuẩn khớp với bộ não AI đã học
class_names = ['ALGAL_LEAF_SPOT', 'ALLOCARIDARA_ATTACK', 'HEALTHY_LEAF', 'LEAF_BLIGHT', 'PHOMOPSIS_LEAF_SPOT']

# Tên hiển thị tiếng Việt cho nông dân dễ đọc
class_names_vi = {
    'ALGAL_LEAF_SPOT': '🍂 Bệnh đốm lá rong biển (Algal Leaf Spot)',
    'ALLOCARIDARA_ATTACK': '🐛 Rầy xanh tấn công (Allocariadara attack)',
    'HEALTHY_LEAF': '🟢 Lá khỏe mạnh - Cây bình thường',
    'LEAF_BLIGHT': '🔥 Bệnh cháy lá / Thối lá (Leaf blight)',
    'PHOMOPSIS_LEAF_SPOT': '🟤 Bệnh đốm lá Phomopsis'
}

# 4. Nạp từ điển cách chữa từ file database.json
try:
    with open('database.json', 'r', encoding='utf-8') as f:
        database = json.load(f)
except Exception as e:
    st.error(f"Lỗi nạp database.json: {e}")
    database = {}

# 5. Khung tải ảnh lên
uploaded_file = st.file_uploader("Tải lên ảnh chụp cận cảnh lá sầu riêng cần kiểm tra:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh người dùng vừa chọn lên màn hình
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh lá cây đã tải lên', use_container_width=True)
    
    with st.spinner('Bác sĩ AI đang soi lá, bạn đợi xíu nha...'):
        # Tiền xử lý ảnh về kích thước 224x224 giống lúc Train
        img_resized = image.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0) # Tạo batch dimension
        
        # Cho AI đoán
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        disease_name = class_names[predicted_index]
        
        # Tính tỷ lệ phần trăm chính xác
        confidence = float(predictions[0][predicted_index] * 100)

    # 6. Xuất kết quả ra màn hình
    st.subheader("📊 Kết quả phân tích:")
    ten_tieng_viet = class_names_vi.get(disease_name, disease_name)
    
    if disease_name == "HEALTHY_LEAF":
        st.success(f"**Tình trạng:** {ten_tieng_viet} (Độ tin cậy: {confidence:.2f}%)")
    else:
        st.warning(f"**Phát hiện bệnh:** {ten_tieng_viet} (Độ tin cậy: {confidence:.2f}%)")
        
    # Xuất cách chữa tương ứng từ database.json
    st.subheader("💡 Hướng dẫn điều trị từ chuyên gia:")
    treatment = database.get(disease_name, "Chưa cập nhật thông tin điều trị.")
    st.info(treatment)
