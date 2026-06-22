import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Cấu hình giao diện
st.set_page_config(page_title="CoffeeCare AI", layout="centered")
st.title("☕ CoffeeCare AI - Chuyên gia Cà phê Đắk Lắk")
st.write("Sử dụng camera điện thoại để quét và nhận diện bệnh trên cây cà phê.")

# 1. Load mô hình Teachable Machine
@st.cache_resource
def load_ai_model():
    model = load_model("keras_model.h5")
    # Đọc file labels
    with open("labels.txt", "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

model, class_names = load_ai_model()

# 2. Kích hoạt Camera điện thoại
#camera_image = st.camera_input("📸 Hướng camera vào lá hoặc cành cà phê cần kiểm tra:")
camera_image = st.file_uploader("Chọn ảnh", type=["jpg", "png"])

# 1. Tạo một mảng trống đúng định dạng của Teachable Machine
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# 2. Đưa ảnh đã chuẩn hóa vào mảng
data[0] = normalized_image_array

# 3. Cho AI dự đoán
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# 4. Hiển thị kết quả chi tiết
st.success(f"Kết quả: {class_name[2:]}") # [2:] để cắt bỏ số thứ tự trong file labels
st.info(f"Độ tin cậy (Confidence): {confidence_score * 100:.2f}%")

if camera_image is not None:
        # 1. Mở và resize ảnh
        image = Image.open(camera_image).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        
        # 2. Chuyển sang mảng
        image_array = np.asarray(image)
        
        # 3. Chuẩn hóa
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        
        # 4. Tạo thùng chứa
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        # 5. Dự đoán
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        
        # 6. Hiển thị kết quả
        disease_name = class_name[2:].strip()
        st.success(f"Kết quả: {disease_name}") 
        st.info(f"Độ tin cậy: {confidence_score * 100:.2f}%")
        
        # 7. Xử lý logic bệnh (Dùng if/elif để không bị trùng lặp)
        if "rust" in disease_name.lower():
            st.error("⚠️ Phát hiện: Bệnh Gỉ Sắt (Coffee Leaf Rust)")
            st.markdown("— **Triệu chứng:** Mặt dưới lá xuất hiện các đốm bột màu vàng cam. Lá bệnh sẽ rụng sớm.")
            st.markdown("— **Cách xử lý:** Cắt tỉa cành sâu bệnh. Phun các loại thuốc gốc Đồng (Copper Hydroxide) để phòng trị.")

        elif "red spider mite" in disease_name.lower():
            st.warning("⚠️ Phát hiện: Nhện Đỏ gây hại (Red Spider Mite)")
            st.markdown("— **Triệu chứng:** Lá mất màu xanh bóng, chuyển sang màu rám bạc hoặc lốm đốm vàng.")
            st.markdown("— **Cách xử lý:** Tưới nước phun áp lực nhẹ để rửa trôi. Sử dụng tinh dầu Neem sinh học.")

        elif "healthy" in disease_name.lower():
            st.success("✅ Không phát hiện bệnh! Lá cà phê khỏe mạnh.")
            st.markdown("— **Lời khuyên:** Cây đang phát triển tốt. Tiếp tục duy trì chế độ chăm sóc hiện tại.")

        else:
            st.info("⚠️ Hệ thống chưa nhận diện rõ. Vui lòng thử chụp lại gần và rõ nét hơn.")
