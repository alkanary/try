import torch
from transformers import VisualBertModel
from myproject.visualbert_tokenizer import BertTokenizer

# تحميل المشفر والمعين
visual_bert = VisualBertModel.from_pretrained("uclanlp/visualbert-base-pre")
tokenizer = BertTokenizer.from_pretrained("uclanlp/visualbert-base-pre") 

# تحميل صورة وتحويلها الى ميزات 
image = "D:/academy/sem7/MiniGPT-4-main/1.png"
image_features = visual_bert.extract_visual_features(image)  

# تحميل نص وعمل تمثيل له
text = "This is an example text."  
inputs = tokenizer(text, return_tensors="pt")

# دمج الميزات البصرية والنصية
inputs["pixel_values"] = image_features
outputs = visual_bert(**inputs)

# الحصول على ميزات التمثيل المشترك للنص والصورة
visual_text_features = outputs.pooler_output