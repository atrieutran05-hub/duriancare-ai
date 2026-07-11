import tensorflow as tf
import numpy as np
import json

# 1. Nạp đúng tên model sầu riêng mới
model = tf.keras.models.load_model('durian_care_model.keras')

# 2. Danh sách 5 bệnh chuẩn của sầu riêng (đã khớp viết hoa viết thường với Colab)
class_names = ['ALGAL_LEAF_SPOT', 'ALLOCARIDARA_ATTACK', 'HEALTHY_LEAF', 'LEAF_BLIGHT', 'PHOMOPSIS_LEAF_SPOT']

# 3. Nạp database cách chữa bệnh
with open('database.json', 'r', encoding='utf-8') as f:
    database = json.load(f)

def predict_and_treat(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    disease_name = class_names[predicted_index]
    
    treatment = database.get(disease_name, "Chưa có thông tin chữa trị cho bệnh này.")
    
    return disease_name, treatment
