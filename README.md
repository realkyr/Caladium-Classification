# Caladium Classification

## _Handcraft Based_
## Configuration
วิธีการนำไฟล์มา Test ให้นำไฟล์ที่จะเทส มาใส่ในโฟลเดอร์ Test

ฟังก์ขั่นที่ใช้ในการ Feature Extraction มีด้วยกัน 4 ฟังก์ชั่นสามารถเลือกใช้อย่างใดอย่างหนึ่ง ได้
- all_feature_extractor
    -   corner_count_feature_extractor  นับจำนวนมุมจากรูป contour
    -   length_width_ratio_feature_extractor สัดส่วนความกว้างยาว
    -   hog_texture_extractor ฮิสโตแกรมของทิศทางเส้น (Texture)
    -   hsv_color_extractor ฮิสโตแกรมปริภูมิสี Hue



- feature_extractor_with_corner_count (ให้ผลลัพธ์รองลงมา)
    -   corner_count_feature_extractor  นับจำนวนมุมจากรูป contour
    -   length_width_ratio_feature_extractor สัดส่วนความกว้างยาว
    -   hsv_color_extractor ฮิสโตแกรมปริภูมิสี Hue


- feature_extractor_only_hue (ให้ผลลัพธ์ดีที่สุด
    -   hsv_color_extractor ฮิสโตแกรมปริภูมิสี Hue

โดยสามารถแก้ไขได้ในไฟล์ ```Utils.py``` โดยการเปลี่ยนชื่อฟังก์ชั่น

## Installation
```sh
pip install -r requirements.txt
```

## Run
```sh
python handcraft_based.py
```
