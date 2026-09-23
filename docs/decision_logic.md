ตอนนี้ต้องการปรับ logic ของ PatchCore ใน VisionAI โดยมี observation สำคัญคือ:

**ผลของ Heatmap ปัจจุบันดูสมเหตุสมผลและตรงกับ defect มากกว่า Segment และผล PASS/FAIL**

ปัญหาปัจจุบันคือบางครั้ง Heatmap แสดงบริเวณ defect ได้ถูกต้อง แต่:

* Segment ไม่ครอบบริเวณที่ Heatmap มองว่า abnormal
* หรือไม่มี Segment เลย
* หรือ PASS/FAIL ยังเป็น PASS ทั้งที่ Heatmap มี local anomaly ชัดเจน

ดังนั้นอยากปรับ architecture ให้ **Anomaly Map เป็นข้อมูลหลัก (single source of truth)** สำหรับ Heatmap, Segment และ local PASS/FAIL decision

## Current flow

ปัจจุบันโดยประมาณ:

```text
PatchCore
   │
   ├── pred_score
   │      ↓
   │   Image threshold
   │      ↓
   │   PASS / FAIL
   │
   └── raw anomaly_map
          │
          ├── Pixel Gate
          │     ↓
          │  hard threshold
          │     ↓
          │  binary mask
          │     ↓
          │  connected components
          │     ↓
          │  min/max area
          │     ↓
          │  Segment / FAIL
          │
          └── display_anomaly_map()
                 ↓
              EMA smoothing
                 ↓
              Heatmap
```

ปัญหาคือ Heatmap และ Pixel Gate ตีความ anomaly map ต่างกันมาก

Pixel Gate ปัจจุบันใช้ hard threshold:

```python
mask = anomaly_map > pixel_threshold
```

จากนั้นใช้ connected components และเช็กพื้นที่ด้วย `min_area_px / max_area_px`

อีกทั้ง `pixel_threshold` ยังมีความสัมพันธ์กับ image threshold ผ่าน `threshold_ratio`

จึงอาจทำให้ข้อมูล local anomaly ที่ยังเห็นชัดใน Heatmap ถูกทิ้งหลัง hard threshold

---

# สิ่งที่ต้องการเปลี่ยน

ต้องการ flow ใหม่ประมาณ:

```text
                 PatchCore
                     ↓
              RAW ANOMALY MAP
                     ↓
          Spatial Processing
                     ↓
          PROCESSED ANOMALY MAP
                ⭐ SOURCE ⭐
              /       |       \
             /        |        \
            ↓         ↓         ↓
        Heatmap    Segment    Local Verdict
```

Heatmap, Segment และ Local Verdict ควรอ้างอิง anomaly representation เดียวกัน เพื่อให้ผลที่แสดงและผลที่ระบบใช้ตัดสินสัมพันธ์กัน

## สำคัญ: ห้ามใช้ Display Heatmap โดยตรงในการตัดสิน

ไม่ต้องการเอาภาพ Heatmap ที่ผ่าน colormap หรือ normalization สำหรับ visualization มาใช้ PASS/FAIL

และไม่ควรใช้ temporal EMA smoothing ข้ามชิ้นงานสำหรับ decision logic

ปัจจุบันมีประมาณ:

```python
smoothed =
    0.6 * previous_map +
    0.4 * current_map
```

EMA แบบนี้สามารถเก็บ anomaly จาก frame/part ก่อนหน้าได้ จึงควรเป็น **display-only**

Decision map ควร deterministic จากชิ้นงานปัจจุบันเท่านั้น

ถ้าต้อง smoothing สำหรับ decision ให้ใช้ spatial smoothing ภายใน anomaly map ของภาพเดียว เช่น Gaussian smoothing ที่เหมาะสม

---

# Segment Logic ที่อยากทดลอง

แทนการใช้ threshold เดียว:

```python
mask = anomaly_map > threshold
```

ให้ทดลอง **dual threshold / hysteresis segmentation**

แนวคิด:

```text
High threshold
→ หา strong anomaly core

Low threshold
→ หา weak anomaly pixels รอบๆ

จากนั้นเก็บ weak pixels เฉพาะส่วนที่เชื่อมกับ strong anomaly core
```

ตัวอย่าง:

```text
Processed anomaly map

   weak weak weak
   weak HIGH weak
   weak HIGH weak

High threshold
→ เจอ core

Low threshold
→ grow region รอบ core

ผลลัพธ์:
Segment ครอบ anomaly region ที่สมเหตุสมผลกว่า hard threshold เดียว
```

จากนั้นค่อยทำ Connected Components

แต่ละ region ควรเก็บข้อมูลอย่างน้อย:

```text
area
peak anomaly
mean anomaly
bounding box / contour
```

ถ้าเป็นไปได้ อาจเพิ่ม local contrast กับบริเวณรอบ region ภายหลัง แต่ version แรกไม่ต้องซับซ้อนเกินไป

---

# PASS/FAIL Logic

ยังไม่ต้องทิ้ง PatchCore image score

ต้องการให้ verdict มีสองแหล่ง evidence:

```text
1. Global anomaly
   → PatchCore image score

2. Local anomaly
   → anomaly region จาก processed anomaly map
```

แนวคิดเริ่มต้น:

```text
FAIL if:

image_score > image_threshold

OR

(
    strong anomaly core exists
    AND
    grown_region_area >= min_region_area
)
```

ภายหลังสามารถใช้ `peak`, `mean`, `area` เพื่อสร้าง region confidence/score ได้ แต่ขอให้ version แรกยัง explainable และ calibrate ง่าย

---

# Threshold ต้องแยกหน้าที่

ไม่ควร assume ว่า:

```text
pixel_threshold =
image_threshold × fixed_ratio
```

เสมอไป

อยากแยก calibration เป็น:

```text
image_threshold
→ calibrate จาก image-level scores

pixel_high_threshold
→ calibrate จาก anomaly-map distribution

pixel_low_threshold
→ calibrate จาก anomaly-map distribution

min_region_area
→ calibrate จาก region statistics
```

เพราะ image score กับ pixel anomaly value มีหน้าที่ต่างกัน

---

# Heatmap กับ Segment ต้องสอดคล้องกัน

เป้าหมาย UX คือ ถ้า Heatmap มี strong local anomaly เช่นบริเวณแดง/ส้มชัดเจน Segment ควรสามารถอธิบายบริเวณเดียวกันได้

เช่น:

```text
Heatmap
      🔴🔴
    🟠🔴🔴🟠
      🟠🟠

Segment
     ┌─────┐
     │     │
     └─────┘
```

Segment contour ควร overlay บนตำแหน่งที่มาจาก decision mask จริง

ไม่ต้องสร้าง Segment จากสีของ Heatmap แต่ทั้งสองต้อง derive จาก anomaly representation เดียวกัน

---

# Debugging goal

หลังปรับแล้วต้องสามารถแยก failure ได้ชัด:

```text
Defect จริง
    ↓
PatchCore anomaly map
    ↓

1. Heatmap ไม่เห็น
   → PatchCore / feature / ROI / resolution problem

2. Heatmap เห็น แต่ Segment ไม่เห็น
   → segmentation/calibration problem

3. Segment ถูก แต่ verdict PASS
   → verdict logic problem
```

นี่เป็นเหตุผลหลักที่ต้องการให้ pipeline มี source กลางที่ชัดเจน

---

# Validation

อย่า replace logic เดิมทันที

ให้สร้าง Decision/Segmentation v2 แล้วเปรียบเทียบกับ current Pixel Gate บน dataset เดียวกัน:

```text
                  PatchCore
                     ↓
                anomaly_map
                /           \
               ↓             ↓
      Current Pixel Gate   Decision v2
               ↓             ↓
            Result A       Result B
                \           /
                  Compare
```

ให้เก็บอย่างน้อย:

```text
image name
ground truth GOOD/DEFECT
defect type
image score
image threshold
pixel high threshold
pixel low threshold
strong peak
region mean
region area
old verdict
new verdict
```

ถ้ามี synthetic PNG defect ที่มี alpha mask ให้ใช้ alpha mask เป็น ground-truth segmentation เพื่อเปรียบเทียบกับ:

* PatchCore anomaly map
* Heatmap
* Segment v2

จะช่วยดูได้ว่า PatchCore เห็น defect แล้ว segmentation ทำข้อมูลหายหรือไม่

---

# Important constraint

การปรับ Decision v2 นี้ **ไม่สามารถแก้ defect ที่ PatchCore anomaly map ไม่เห็นตั้งแต่ต้น**

ดังนั้นเป้าหมายของงานนี้คือ:

> ทำให้ Segment และ PASS/FAIL ใช้ information ที่ PatchCore anomaly map มีอยู่แล้วได้ดีขึ้น และสอดคล้องกับ Heatmap ที่ปัจจุบันให้ผลน่าพอใจ

หลังจาก Decision v2 เสถียรแล้ว ค่อยแยกงานอีกส่วนสำหรับ tiny defects ที่ Heatmap ยังไม่สามารถ detect ได้ เช่น ROI, input resolution, feature layers หรือ region-specific inspection

กรุณาตรวจ implementation ปัจจุบันก่อนแก้ โดยเฉพาะ:

* `services/patchcore_service.py`
* `utils/pixel_gate.py`
* `services/verdict_service.py`
* calibration logic
* จุดที่สร้าง/render Heatmap
* จุดที่สร้าง segment/contour สำหรับ UI

และพยายาม reuse current raw anomaly map pipeline แทนการสร้าง inference path ใหม่
