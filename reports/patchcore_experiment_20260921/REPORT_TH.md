# ผลทดลอง PatchCore จาก Recipe TJA1041_SO14

ทดลองวันที่ 21–22 กันยายน 2026 — ยังไม่ได้เปลี่ยนโมเดลหรือ Recipe ที่ใช้งานจริง

## สรุปสำหรับการตัดสินใจ

**256 เร็วกว่า แต่ไม่ใช่การลดขนาดที่ได้คุณภาพเท่าเดิม** ในการทดลองนี้ 256 / bank 512 ตรวจพบรอยจำลอง 61/84 ภาพ ขณะที่ 512 / bank 512 พบ 76/84 ภาพ โดยรอยขีดข่วนยังเป็นจุดอ่อนทั้งสองขนาด ดังนั้นไม่แนะนำให้กลับไป 256 แบบเงียบ ๆ หากงานต้องตรวจรอยเล็ก

ตัวเลือกสำหรับนำไปพัฒนาต่อคือ **512 × 512, wide_resnet50_2, layer2 + layer3, bank 512, num_neighbors=1, FP32** โดยใช้วิธีสร้าง bank ตามสคริปต์ทดลองนี้ ไม่ใช่เพียงแก้ coreset ratio ใน Trainer เดิม ตัวเลือกนี้ผ่านเกณฑ์คัดกรองเบื้องต้นของการทดลอง: calibration ตรวจรอยจำลองได้อย่างน้อย 80% และไม่ปฏิเสธภาพดีใน calibration แล้วเลือกตัวที่เร็วที่สุดในกลุ่มนั้น เกณฑ์ 80% นี้เป็นเพียง screening ไม่ใช่มาตรฐานอนุมัติ production

ผลเทียบโมเดลเดิมจริง: ทั้งสองตรวจพบ synthetic 76/84 และ false rejects 0/21 รอบ runtime ที่ 4 threads ลดจาก 8.16 เป็น 2.37 วินาที/ชิ้น และที่ 8 threads ลดจาก 5.31 เป็น 1.35 วินาที/ชิ้น ประมาณ 3.4–3.9 เท่า **ยังไม่ใช่ผลตรวจแบบทันที** และเวลารอบ runtime ช้ากว่า main sweep อย่างชัดเจน จึงห้ามนำตัวเลขเร็วที่สุดไปสัญญาเป็นความเร็วใช้งานจริง

**ยังไม่มีหลักฐานเพียงพอให้รับรอง production หรือความแม่นยำกับของเสียจริง** และยังไม่ได้ calibrate pixel/area gate ให้โมเดลทดลอง จึงยังไม่ควรนำ bank ทดลองไปทับโมเดลเดิม

## ข้อมูลและวิธีทดลอง

- ใช้ prepared dataset ของ Recipe เดิม: train good 100, calibration good 21, test good 21 ภาพ ไม่เปลี่ยน crop หรือแบ่งข้อมูลใหม่
- Audit ไม่พบภาพซ้ำแบบ decoded-image hash ข้าม split และตรวจว่า dataset ก่อน–หลังตรงกัน อย่างไรก็ตาม ยังไม่รับรองว่าไม่มีภาพคล้ายกันจากชิ้นงานเดียวกันหรือ burst เดียวกัน
- Dataset fingerprint: `f620c3a45d693b1eb2bcc5485e77d554185fa9ec86a35282607bf61e416a6db5`
- สร้าง scratch, spot, chip และ pin_discoloration อย่างละหนึ่งภาพต่อภาพดี: calibration synthetic 84 และ test synthetic 84 ภาพ ภาพเสียจำลองไม่ถูกใส่ใน train bank
- ทุก configuration หา threshold จาก calibration และบันทึกก่อนประเมิน test ของขนาดนั้น ไม่ใช้ test ปรับ threshold
- เครื่องทดลอง CPU Intel Core i5-12450H, RAM ประมาณ 16 GB, PyTorch 2.13.0+cpu ไม่มี CUDA; main sweep ใช้ 4 intra-op threads, FP32, inference batch 1
- ใช้ pretrained backbone จากไฟล์ local `models/backbone/wide_resnet50_2/model.safetensors` ไม่ใช่ random backbone; `pre_trained=False` ในสคริปต์ใช้เพื่อไม่ให้ดาวน์โหลด แล้วโหลด weights เอง
- Resize ตาม preprocessor ของ Anomalib และ ImageNet normalization; การทดลองนี้ใช้ภาพ prepared ที่มีอยู่ ไม่ได้ทดสอบ capture/YOLO crop สดครบ flow

### วิธีสร้าง memory bank ต้องอ่านก่อนนำผลไปใช้

เป็นวิธีทดลองสองขั้นเพื่อควบคุม RAM/เวลา: สุ่ม 128 patch ต่อภาพจากภาพ train ทั้ง 100 ภาพ ได้ candidate 12,800 vectors แล้วใช้ Gaussian projection 32 dimensions และ farthest-first เลือก 2,048 vectors; ใช้ 512 ตัวแรกเป็น bank เล็ก Seed = 20260921

ไม่ใช่ native coreset ของ Anomalib ที่ทำกับ patch ทั้งหมด จำนวน patch เต็มที่ 256/384/512 คือ 102,400 / 230,400 / 409,600 ตามลำดับ ดังนั้น **bank 512 ไม่ได้หมายถึงตั้ง `coreset_sampling_ratio=0.005` แล้วจะได้ผลเดียวกัน** และผลไม่ได้พิสูจน์ว่า bank เล็กดีกว่า bank ใหญ่ในทุก dataset

## ผลเปรียบเทียบ 12 configurations

ผลตรวจตารางนี้ใช้ image-score threshold เท่านั้น ไม่รวม pixel gate ของโมเดลทดลอง ทุกตัวมี false reject ของภาพดี test = 0/21

เวลาเป็น median และ empirical p95 จาก 8 ภาพ good test ต่อ configuration รวม preprocessing + backbone + nearest-neighbor + anomaly map ไม่รวมอ่านไฟล์, capture, YOLO, marking, UI และการประมวลผลหลายชิ้นพร้อมกัน จึงไม่ใช่ FPS ของโปรแกรมจริง

| Input | Bank | Neighbors | Threshold | Calibration synthetic | Test synthetic | Median ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | 512 | 1 | 0.628634 | 48/84 | 61/84 (72.6%) | 160.9 | 179.2 |
| 256 | 512 | 9 | 0.628141 | 48/84 | 61/84 (72.6%) | 160.2 | 174.3 |
| 256 | 2048 | 1 | 0.626421 | 44/84 | 52/84 (61.9%) | 199.3 | 218.4 |
| 256 | 2048 | 9 | 0.622946 | 44/84 | 52/84 (61.9%) | 187.9 | 203.4 |
| 384 | 512 | 1 | 0.646161 | 55/84 | 69/84 (82.1%) | 408.8 | 434.9 |
| 384 | 512 | 9 | 0.646159 | 54/84 | 69/84 (82.1%) | 395.6 | 421.1 |
| 384 | 2048 | 1 | 0.637082 | 53/84 | 66/84 (78.6%) | 461.7 | 470.9 |
| 384 | 2048 | 9 | 0.633975 | 54/84 | 66/84 (78.6%) | 449.1 | 471.7 |
| 512 | 512 | 1 | 0.637395 | 69/84 | 76/84 (90.5%) | 666.6 | 705.9 |
| 512 | 512 | 9 | 0.637196 | 69/84 | 76/84 (90.5%) | 676.8 | 700.5 |
| 512 | 2048 | 1 | 0.621480 | 69/84 | 75/84 (89.3%) | 759.9 | 778.6 |
| 512 | 2048 | 9 | 0.618928 | 70/84 | 76/84 (90.5%) | 762.4 | 805.2 |

ความต่างเล็กน้อยระหว่าง neighbors 1 และ 9 ไม่ควรตีความว่า 9 เร็วกว่าโดยธรรมชาติ: วัดเพียง 8 ภาพและลำดับการรัน/อุณหภูมิ CPU มีผล ครั้งนี้ยังไม่เห็นประโยชน์ชัดเจนต่อจำนวน test defects ที่ตรวจพบจากการเพิ่มเป็น 9

### แยกชนิดรอยจำลอง: bank 512, neighbors 1

| Input | Chip /21 | Edge discoloration /21 | Scratch /21 | Spot /21 |
|---|---:|---:|---:|---:|
| 256 | 16 | 14 | 11 | 20 |
| 384 | 21 | 17 | 11 | 20 |
| 512 | 21 | 20 | 14 | 21 |

512 ยังพลาด scratch 7/21 ภาพ ไม่ใช่โมเดลที่ตรวจรอยได้ครบ ความเร็ว 256 bank512 ประมาณ 4.1 เท่าของ 512 bank512 ใน main sweep แลกกับการตรวจพบลดลง 15 ภาพจาก 84 ภาพจำลอง ส่วน 384 เป็นทางประนีประนอม แต่ไม่ผ่าน calibration screening 80% ของการทดลองนี้

## เทียบโมเดลเดิมที่บันทึกไว้

Baseline คือ `recipes/TJA1041_SO14/model_versions/c59dc39a1b6a44b4be0c682003bfb0e8` ซึ่งมี bank shape `[32768, 1536]` ใช้ threshold ของ Recipe 0.5954712295532226 ไม่ได้นำ threshold ของ candidate ไปแทน

โหลด weights ของ baseline แล้วตรวจ feature-extractor tensors ว่าตรงกับ local pretrained backbone ที่ใช้สร้าง candidate ทุกตัว Baseline รัน input512/layer2+layer3/k9 ตาม configuration ปัจจุบัน แต่ metadata เก่าไม่ได้บันทึก input size/layers/k จึงยังยืนยัน configuration ตอนสร้าง artifact จาก metadata เพียงอย่างเดียวไม่ได้

Runtime comparison รันแยกจาก main sweep และใช้ภาพ good 8 ภาพชุดเดียวกัน พร้อม warmup 1 ครั้งต่อ thread setting ควรเทียบเวลาจากรอบ runtime เดียวกันเป็นหลัก เพราะสภาพเครื่องต่างเวลาอาจต่างกันมาก ตัวเลขนี้เป็น single-model offline inference ไม่ใช่ latency ของ live application

| Threads | Baseline median ms | Baseline p95 ms | Candidate median ms | Candidate p95 ms | อัตราส่วน median baseline/candidate |
|---:|---:|---:|---:|---:|---:|
| 2 | 17785.9 | 19133.4 | 4563.3 | 5139.2 | 3.90× |
| 4 | 8162.6 | 9062.6 | 2372.6 | 2656.9 | 3.44× |
| 8 | 5312.5 | 5736.3 | 1351.5 | 1595.2 | 3.93× |

Candidate512/bank512/k1 ที่ 4 threads ใน main sweep วัดได้ 666.6 ms แต่รอบ runtime วัดซ้ำได้ 2372.6 ms เป็นความแปรผันที่มีนัยต่อการใช้งาน ไม่ได้เก็บ CPU clocks, power mode, temperature และ background load ครบ จึงยังยืนยันสาเหตุไม่ได้ และไม่ควรอ้างว่าเป็น thermal throttling โดยไม่มีหลักฐาน อัตราส่วนในตารางเป็นผลของการวัดตามลำดับครั้งนี้ ไม่ใช่ randomized benchmark ที่ควบคุมสภาพเครื่องทั้งหมด

Baseline ประเมินใหม่ครบ good21 + synthetic84 ด้วย threshold เดิม: image rule พบ 76/84 และ full verdict รวม pixel gate พบ 76/84; good ถูกปฏิเสธ 0/21 ทั้งสองกฎ แยกชนิดคือ chip21, edge discoloration20, scratch14, spot21 ซึ่งจำนวนตรงกับ candidate512/bank512/k1 แบบ image rule ไม่ได้อาศัย acceptance report เก่ามาสรุปผลใหม่

ตรวจเทียบรายภาพด้วยแล้ว image PASS/FAIL ของ baseline และ candidate ตรงกันทั้ง 105 ภาพของชุดนี้ ไม่ใช่เพียงจำนวนรวมบังเอิญเท่ากัน แต่ยังไม่รับรองว่าจะตรงกันกับข้อมูลนอกชุดทดสอบ

ข้อจำกัดการเทียบ: baseline ใช้ pixel gate ที่ calibrated แล้ว ส่วน candidate ยังเทียบ image rule เท่านั้น จึงไม่ใช่การอนุมัติ replacement แบบ end-to-end

ยังไม่ควรตั้งทั้งโปรแกรมเป็น 8 threads จากตารางนี้ทันที เพราะ inspection มี YOLO และงาน UI ทำพร้อมกัน ต้องวัด CPU contention และ latency ของผลตรวจสดด้วย

ถ้าต้องการผลจริงต่อชิ้นต่ำกว่าหลักร้อยมิลลิวินาที การลด bank อย่างเดียวบน configuration CPU ที่วัดซ้ำนี้ยังไม่ถึงเป้า ควรทดลอง GPU หรือ backbone ที่เบาลงเป็นงานถัดไป พร้อมทดสอบคุณภาพใหม่ ไม่ใช่ซ่อนเวลา inference ด้วยสถานะ DETECTED

## Threshold: ปัญหาไม่ได้อยู่แค่ตัวเลขเดียว

ฟังก์ชันปัจจุบันเลือก threshold ที่จับ synthetic ได้มากสุดภายใต้ false alarms บน calibration good ไม่เกิน 1% แต่เมื่อมีเพียง 21 ภาพ เกณฑ์นี้เท่ากับต้องไม่พลาดภาพดีเลย ทำให้ภาพดีที่ score สูงที่สุดมีอิทธิพลมาก

คะแนนที่รายงานคือ raw PatchCore distance score หารด้วย 100 ตาม convention ของโปรเจค ไม่ใช่ probability หรือเปอร์เซ็นต์ความมั่นใจ และไม่ได้รับประกันว่าจะอยู่ในช่วง 0–1 เสมอ

ตัวอย่าง 256 / bank512 / k1: calibration good สูงสุด `Image_28419.jpg` = 0.627655, อันดับถัดไป 0.544181 และ threshold = 0.628634 ภาพสูงสุดมีความต่างด้านความสว่าง/มุมที่เห็นได้จากการตรวจภาพ จึงควรตรวจสภาพการถ่ายและเพิ่มตัวอย่างปกติที่ครอบคลุมความแปรผัน ไม่ควรลบภาพนั้นหรือปรับ threshold ลงเพื่อให้คะแนนสวยโดยไม่ยืนยันว่าภาพเป็น good จริงหรือไม่

Synthetic ROC AUC ของ configurations สูงประมาณ 0.993–0.999 แต่ recall ณ threshold ที่ใช้งานต่ำกว่าได้มาก จึงไม่ควรใช้ AUC ตัวเดียวรับรองผล PASS/FAIL

ข้อเสนอรอบต่อไป:

1. แยกภาพ train, calibration และ acceptance ใหม่จากคนละช่วงถ่าย/ล็อตเมื่อทำได้ โดยรักษา crop ให้ตรง Inspection
2. Calibrate ทั้ง image score และ pixel threshold/connected-component area สำหรับแต่ละ input size ไม่คัดลอก threshold หรือ area 64 px ไปใช้ข้ามขนาดทันที
3. ถ้าจะแยกขาหรือบริเวณตัวถัง ให้ทดลอง ROI/region-specific score เพิ่มก่อน ไม่จำเป็นต้องเริ่มจาก detector ขาทีละขา แต่ต้องมี alignment ที่เชื่อถือได้และ labels สำหรับประเมิน
4. ไม่ normalize anomaly map ตาม min/max ของแต่ละภาพแล้วใช้ตัดสินแทนเกณฑ์ calibrated เพราะอาจขยาย noise ของภาพดีหรือกลบภาพที่เสียทั้งบริเวณ; ใช้ normalization แบบนั้นเพื่อแสดงผลได้ แต่เกณฑ์ตัดสินต้องมี reference ที่คงความหมาย

## Hyperparameters ที่แนะนำให้ทดลองใช้งานต่อ

เหตุผลที่ bank มีผลมาก: ที่ input512 feature grid มี 4,096 patches และ baseline bank32,768 ทำให้มีคู่ระยะห่างประมาณ 134 ล้านคู่ต่อชิ้น เทียบกับประมาณ 2.1 ล้านคู่เมื่อ bank512 การลด bank จึงลดงานค้นหาได้มากโดยไม่ลดความละเอียด input แต่เวลา backbone ยังอยู่ และการตัด representative features มากเกินไปอาจทำให้คุณภาพลดลง ต้องวัดพร้อมกันเสมอ ไม่ได้หมายความว่าโปรแกรมจะเร็วขึ้น 64 เท่า

| Parameter | ค่า/แนวทาง |
|---|---|
| Input size | 512 × 512 สำหรับรักษารายละเอียด; 256 เป็นโหมดเร็วที่ต้องยอมรับและยืนยันความเสี่ยงรอยหลุด |
| Backbone | wide_resnet50_2 pretrained ตัวเดียวกับปัจจุบัน; ยังไม่ได้เทียบ backbone อื่น |
| Feature layers | layer2 + layer3; ยังไม่ได้ ablation layers |
| Bank | 512 vectors ด้วยวิธีสองขั้นในสคริปต์นี้ เป็น candidate ไม่ใช่ค่าการันตี |
| num_neighbors | 1 สำหรับ candidate ที่เลือก; 9 ไม่เพิ่มจำนวน test detections ใน bank512 ครั้งนี้ |
| Precision/device | FP32 CPU ตามที่วัด; ยังไม่ได้ทดสอบ GPU/FP16/quantization |
| Inference batch | 1 ตามการทดลอง; การ batching หลาย ROI ต้องวัด latency และ RAM ใหม่ |
| Sampling | 128 candidates/image, projection32, seed20260921, farthest-first; ต้องบันทึกวิธีพร้อม bank |
| Image threshold | 0.6373954391479493 เฉพาะ candidate512/bank512/k1 นี้ ห้ามนำไปใช้กับโมเดลเดิม |
| Pixel/area gate | ต้อง calibrate ใหม่ก่อน deploy; ยังไม่ให้ค่า production |
| Epoch/LR/optimizer | ไม่ใช่ตัวปรับหลักของ PatchCore แบบ frozen pretrained features; สร้าง bank จาก features หนึ่งรอบ ไม่ใช่ gradient training |

เวลา feature extraction + สร้าง bank ถึง 2048 ร่วมกัน: 256 = 17.35 s, 384 = 37.58 s, 512 = 64.33 s ไม่รวม import, calibration, synthetic generation และ evaluation และไม่ใช่เวลาเทรน bank512 ที่วัดแยกโดยเฉพาะ

ก่อนรองรับ candidate ใน Trainer ควรบันทึก image_size, backbone/weights hash, layers, neighbors, precision, sampling method/seed, จำนวน bank จริง และ calibration policy ไว้ใน model metadata แล้วให้ Inspection อ่านค่าที่ตรงกัน ไม่พึ่ง global input size อย่างเดียว

### ทำให้ผู้ใช้เทรนง่ายโดยไม่ต้องเดาค่า

ข้อเสนอสำหรับรอบ implementation ถัดไป ไม่ได้แก้ UI/Trainer ในงานทดลองนี้:

1. ผู้ใช้ใส่ภาพดีและเลือกประเภทงาน/รายละเอียดตำหนิที่ต้องตรวจ ระบบตรวจภาพและเตือนการถ่ายที่ต่างกันมากก่อนเริ่ม
2. ระบบใช้ crop ตาม Inspection แล้วสร้าง train/calibration/acceptance ที่แยกกัน ผู้ใช้ไม่ต้องกำหนด threshold เอง
3. ทดลอง preset จำนวนจำกัด เช่น 512 + bank caps หลายระดับ แล้วเลือกจาก calibration sensitivity และเวลาที่วัด ภายใต้เงื่อนไข false rejects; ไม่เลือกจาก test score ที่ดีที่สุดย้อนหลัง
4. ถ้าไม่มี preset ผ่านเกณฑ์ ให้แสดงว่าต้องเพิ่มข้อมูล/ปรับสภาพถ่าย ไม่แอบลดเกณฑ์เพื่อขึ้น Ready
5. ประเมิน acceptance อิสระหนึ่งครั้ง พร้อมแสดงจำนวนภาพดีที่ถูกปฏิเสธและรอยที่พลาดแยกชนิด; ถ้ามีแต่ synthetic ให้ระบุสถานะว่ายังไม่ยืนยันของเสียจริง
6. บันทึกโมเดลพร้อม preprocessing/calibration metadata แล้วทดสอบ reload และเทียบผลกับ Inspection ก่อนอนุมัติใช้จริง

## ข้อจำกัดและเงื่อนไขก่อน production

- ไม่มีภาพของเสียจริง: ตัวเลข detection ในรายงานเป็นรอยจำลองเท่านั้น ไม่ใช่ความแม่นยำของงานจริง
- `pin_discoloration` ใน generator วางสี่เหลี่ยมสีบน edge band ไม่ได้รู้ว่าขาอยู่ตรงไหน อาจลงบนพื้นหลัง จึงเรียกว่า edge discoloration ในผล ไม่ใช่การพิสูจน์ว่าตรวจสีที่ขาได้
- รอยจำลองหลายแบบมาจากภาพต้นทางเดียวกัน มีความสัมพันธ์กัน ไม่ใช่ของเสียอิสระ 84 ชิ้น
- 0 false rejects จาก good 21 ภาพไม่ได้พิสูจน์ว่าอัตราพลาดต่ำกว่า 1%: ขอบบน one-sided 95% อยู่ประมาณ 13.3% ภายใต้สมมติฐานตัวอย่างอิสระ ต้องมีอย่างน้อย 299 ภาพ good อิสระที่ไม่พลาดเลยจึงเริ่มรองรับขอบบน 1% แบบเดียวกัน
- หลังเห็นผล test ของ 12 configurations แล้ว ชุดนี้เป็น exploratory benchmark ไม่ควรใช้ซ้ำเป็น acceptance สุดท้าย ต้องเตรียม fresh holdout
- ใช้เพียงหนึ่ง seed, หนึ่ง recipe, เครื่อง CPU เครื่องเดียว และ timing ตัวอย่างน้อย ยังไม่วัด soak test/thermal/multi-object concurrency/GUI latency
- ยังไม่ได้เปรียบเทียบ backbone เล็ก, native full-pool coreset, GPU, feature compression หรือค้นหา hyperparameters ทุกแบบ คำว่า candidate ที่เหมาะหมายถึงในขอบเขตที่ทดลอง ไม่ใช่ดีที่สุดเท่าที่เป็นไปได้

## ไฟล์ผลและวิธีรันซ้ำ

- `results.json`: คะแนนต่อภาพ, calibration, thresholds, metrics และ latency samples ของทั้ง 12 configurations
- `summary.csv`: ตารางสรุปที่สร้างและตรวจจำนวนผลจาก `results.json` ด้วย `tools/summarize_patchcore_experiment.py`
- `frozen_calibration_256.json`, `frozen_calibration_384.json`, `frozen_calibration_512.json`: calibration ที่บันทึกก่อน test
- `runtime.json`: ผลเทียบ saved baseline กับ candidate และ CPU threads (ดูส่วน runtime ประกอบ)
- `../../tools/benchmark_recipe_patchcore.py`: รัน main sweep
- `../../tools/benchmark_patchcore_runtime.py`: รัน baseline comparison หลัง main sweep เสร็จ
- `bank_*.pt` และ synthetic folders เป็น artifacts ทดลองที่ exclude จาก Git ด้วย `.gitignore` ไม่ใช่ deployment bundle

รันจาก project root โดยใช้ Python environment ที่มี dependencies ของโปรเจค:

```powershell
python -u tools/benchmark_recipe_patchcore.py
python -u tools/benchmark_patchcore_runtime.py
```

อย่ารันสองสคริปต์พร้อมกันหรือเปิด inference หนักอื่นระหว่างวัดเวลา ผลลัพธ์จะเขียนทับเฉพาะโฟลเดอร์รายงานเดิม ไม่ได้เปลี่ยน active recipe/model

## อ้างอิงหลักการ

PatchCore ใช้ pretrained patch features และ representative memory bank ไม่ใช่การเทรน classifier ด้วย epochs/LR ตามปกติ: [Anomalib PatchCore reference](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/patchcore.html), [บทความต้นฉบับ Towards Total Recall in Industrial Anomaly Detection](https://www.amazon.science/publications/towards-total-recall-in-industrial-anomaly-detection)

ตัวเลขในรายงานนี้มาจากการรันข้อมูล Recipe ในเครื่อง ไม่ได้นำ benchmark สาธารณะมาอ้างเป็นความแม่นยำของโปรเจค

## การตรวจสอบก่อนส่งรายงาน

- Main sweep มี 12 configurations ครบ; ตรวจ threshold/rows กับ frozen calibration และคำนวณ test counts ซ้ำจากคะแนนรายภาพแล้วตรงกัน
- Runtime process จบ exit code 0; มี 6 timing configurations และ baseline evaluation ครบ 105 ภาพ
- คะแนน candidate จาก full-model forward ตรงกับวิธีแยก feature/search ใน sweep ภายใน tolerance 1e-5 ทุก thread setting ที่วัด
- Baseline verdict ไม่มีเหตุผลประเภท unavailable/invalid และประเมินกับ test filenames เดียวกัน
- Dataset audit หลังทดลองตรงกับก่อนทดลอง; SHA256 ของ weights และ memory bank เดิมตรงกับ metadata
- สคริปต์ผ่าน syntax compilation และ `git diff --check` ไม่มี whitespace errors; นี่เป็นการตรวจเสริม ไม่ใช่หลักฐานแทนการรันโมเดลจริงข้างต้น
- ใช้แนวทางจากสกิล deep-learning-pytorch แยก calibration/test และบันทึก seed; verification-before-completion ใช้ตรวจผลดิบก่อนสรุป ไม่ได้ใช้สกิลเหล่านี้แก้ Trainer/Inspection
