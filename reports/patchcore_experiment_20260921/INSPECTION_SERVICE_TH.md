# ทดลอง bank เล็กผ่าน PatchCoreService — 23 กันยายน 2026

ใช้ service ที่ Inspection เรียกจริง โหลดโมเดลเดิมแล้วสลับ bank เฉพาะใน process ทดลอง และคืนค่าเดิมเมื่อเสร็จ ไม่แก้ active Recipe หรือ weights บนดิสก์

| CPU threads | เดิม bank32768 / k9 | ทดลอง bank512 / k1 | อัตราส่วนเวลา |
|---|---:|---:|---:|
| 4 | 2526 ms | 703 ms | 3.60 เท่า |
| 8 | 2056 ms | 573 ms | 3.59 เท่า |

เป็น median ของภาพ good 4 ภาพชุดเดียวกัน หลัง warmup รวม preprocessing, inference และ heatmap ผ่าน service จริง ไม่รวม capture, YOLO, orientation/marking, segment หรือ Qt paint ไม่มีการรัน YOLO พร้อมกัน จึงยังไม่ใช่ latency ทั้งโปรแกรม ตัวอย่างน้อยและลำดับการรันอาจได้รับผลจากสภาพ CPU

ผลบอกว่าการลด bank ช่วยจริง แต่ 573 ms ต่อชิ้นยังประมาณ 1.7 ผลตรวจต่อวินาทีในขอบเขตนี้ และหลายชิ้นจะช้าลง ไม่ใช่ภาพเคลื่อนไหวพร้อมผลที่ 10–30 ครั้งต่อวินาที

Candidate ผ่าน calibration และ independent synthetic acceptance เมื่อ 23 กันยายน 2026 แล้ว Recipe `TJA1041_SO14` ใช้ model version `f356ef620e0f4fe887b01f0783f8c4b4`, image threshold `0.6373954391479493`, pixel gate ratio `1.0`, min area `1 px` และ `num_neighbors=1` จาก metadata ของ artifact ทุกค่าเป็นของ candidate ไม่ใช้ pixel gate ของโมเดลเดิม

ผล acceptance: good test 0/21 false rejects, synthetic image rule 76/84 (90.5%), synthetic pixel gate 74/84 (88.1%), pixel gate good 0/21. Calibration pixel gate จับ synthetic 64/84 (76.2%) จึงยังต้องเก็บของเสียจริงเพิ่มเติมก่อนรับรองความแม่นยำสำหรับ production

ดูผลคะแนน/เวลาทุกรอบและเวลา load_model ใน `inspection_service_runtime.json` เวลา load_model ไม่รวม Python/module imports ก่อนสร้าง service และยังไม่ได้แยก cold disk cache

ขั้นต่อไปสำหรับการใช้งาน: calibrate pixel gate ของ candidate และวัดทั้ง pipeline ภายใต้ YOLO พร้อมกัน หากต้องการผลพร้อมกรอบเร็วกว่า 0.1–0.2 วินาที ต้องทดลอง backbone ที่เบาลงหรือ GPU และทดสอบคุณภาพใหม่ ผลครั้งนี้ยังไม่ยืนยันว่ารักษาความแม่นยำที่ความเร็วนั้นได้

รันซ้ำ: `python -u tools/benchmark_inspection_service.py`
