import tensorflow as tf
import numpy as np
import json

# 1. Nạp model
model = tf.keras.models.load_model('coffee_care_model.keras')

# 2. Danh sách tên bệnh (Dán cái list ní vừa copy ở Colab vào đây)
class_names = ['Apple_Alternaria_Leaf_Spot', 'Apple_Brown_Spot', ...] # Điền đầy đủ list của ní vào

# 3. Nạp database cách chữa bệnh
with open('database.json', 'r', encoding='utf-8') as f:
    database = json.load(f)

def predict_and_treat(image_path):
    # Xử lý ảnh
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    # AI dự đoán
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    disease_name = class_names[predicted_index]
    
    # Lấy thông tin từ database
    # Lưu ý: Tên key trong database.json phải trùng khớp với tên trong class_names
    treatment = database.get(disease_name, "Chưa có thông tin chữa trị cho bệnh này.")
    
    return disease_name, treatment

# --- CÁCH DÙNG ---
# result_name, info = predict_and_treat('ảnh_cây_cần_khám.jpg')
# print(f"AI đoán là: {result_name}")    
# print(f"Cách chữa: {info}")
