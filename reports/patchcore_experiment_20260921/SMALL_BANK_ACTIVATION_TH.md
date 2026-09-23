# เปิดใช้ PatchCore bank 512 — 23 กันยายน 2026

Recipe `TJA1041_SO14` ใช้ model version `f356ef620e0f4fe887b01f0783f8c4b4` แล้ว

| รายการ | ผล |
|---|---:|
| Memory bank | 512 × 1536 vectors |
| Neighbors | 1 |
| Input | 512 × 512 |
| Image threshold | 0.6373954391479493 |
| Pixel threshold | 63.7395 raw-map units; heatmap ช่วงเหลืองถึงส้ม |
| Pixel min area | 1 px |
| Good test false rejects | 0/21 |
| Synthetic image detection | 76/84 (90.5%) |
| Synthetic pixel detection | 74/84 (88.1%) |

การตรวจ acceptance ใช้ calibration เพื่อเลือก threshold/pixel gate แล้วทดสอบกับ test split ที่แยกไว้ โดยไม่มีภาพของเสียจริง ดังนั้นตัวเลข synthetic ใช้ยืนยันเพียงพฤติกรรมของรอยจำลอง ไม่ใช่ความแม่นยำของของเสียจริง

โมเดลเดิมยังอยู่ใน `model_versions/c59dc39a1b6a44b4be0c682003bfb0e8` และ Recipe สามารถชี้กลับได้หากต้องเปรียบเทียบอีกครั้ง
