## ชื่อกลุ่ม อ่างไม่ลง องค์จะลงได้ไง
## รายชื่อสมาชิกกลุ่ม
- 1.นายบดินทร์ หนูรัก สาขา IT 60070043
- 2.นายภูรี กานุสนธิ์ สาขา IT 60070075
- 3.นายชนาธิป นวลศรี สาขา DSBA 62070235
- 4.นายภณ พงษ์วชิรินทร์ สาขา DSBA 62070262
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
## _Learning Based_
## Configuration
วิธีการนำไฟล์มา Test ให้นำไฟล์ที่จะเทส มาใส่ในโฟลเดอร์ Test
วิธีการนำไฟล์ Test มาใช้ จะต้องทำการ run ไฟล์ handcraft_based.py ก่อนเพื่อให้ Export ข้อมูล test และ train ออกมาเป็น csv
## Installation
```sh
pip install -r requirements.txt
```

## Run
```sh
python learning_based.py
```
## Configuration
ไฟล์ learning_based.py เป็นไฟล์ที่เอาไว้ Train และ Save model 

ไฟล์ learning_based_test.py เป็นการ Load Model มาทดสอบ และ เทสเพื่อดูผลลัพธ์


ถ้ามี Error path ให้ทำการแก้ path ของไฟล์ csv ด้วยตัวเอง
และถ้าเกิดไฟล์ไม่สามารถ run ได้ ให้ใช้ไฟล์ learning_based_test.ipynb รันผ่าน Google colab 
และ ทำการ Upload ไฟล์ csv ขึ้นระบบด้วยตัวเอง และ copy path ด้วยตัวเอง
