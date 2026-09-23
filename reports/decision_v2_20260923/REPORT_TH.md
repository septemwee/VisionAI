# ผลทดลอง Decision / Segmentation v2

ยังไม่เปิดใช้ใน Inspection: Recipe ยังคงใช้ `image_and_pixel_v1` ไม่มีการเปลี่ยนโมเดลหรือเกณฑ์ที่ใช้งานจริงจากการทดลองนี้

## วิธีทดลอง

- ใช้โมเดลปัจจุบันของ TJA1041_SO14: input 512, memory bank 512, neighbors 1
- ใช้ calibration ของดี 21 ภาพ + รอยจำลอง 84 ภาพ เลือกเกณฑ์แล้วตรึงค่า ก่อนประเมิน test อีก 21 + 84 ภาพ
- แต่ละภาพรัน PatchCore ครั้งเดียว ส่ง raw map เดียวกันให้ v1 และ v2
- v2 ทำ Gaussian spatial smoothing sigma 0.8 ในภาพเดียว ไม่มี temporal EMA
- ใช้ high threshold หาแกนรอย และเก็บพื้นที่เหนือ low threshold ที่เชื่อมกับแกนด้วย 8-connectivity
- Heatmap v2 ใช้ processed map; Segment และ local verdict ใช้ mask จาก processed map เดียวกัน
- เลือกเกณฑ์ local จากสถิติ map ชุด calibration โดยไม่ผูกกับ image threshold เลือกให้ตรวจพบ synthetic มากที่สุดภายใต้ของดีถูกตัดตก 0 ภาพ
- Candidate grid มี high threshold จาก quantile ของ peak ของดี, low จาก quantile ของค่าพิกเซลของดี และ minimum area หลายค่า เป็นการค้นหาเบื้องต้น ไม่ใช่ global optimum
- Verdict = image score เกินเกณฑ์ OR มี region ผ่านเกณฑ์พื้นที่
- Alpha mask ของรอยจำลอง > 0.1 ใช้เป็น ground truth เชิงทดลอง ย่อ/ขยายสู่ขนาด map ก่อนวัด
- บันทึก identity โมเดลและ audit dataset ใน JSON และตรวจว่าไม่มีการเปลี่ยนระหว่างรัน

## ค่าที่เลือกจาก calibration

| ค่า | ผล |
|---|---:|
| Image threshold เดิม | 0.6373954391 |
| Pixel high (raw distance) | 62.9089088440 |
| Pixel low (raw distance) | 57.9124272728 |
| Minimum grown region area | 1 map pixel |
| Spatial sigma | 0.8 |

## ผล held-out test

| ตัวชี้วัด | v1 | v2 |
|---|---:|---:|
| ของดีถูกตัดตก | 0/21 | 0/21 |
| ตรวจพบภาพ synthetic จาก image OR local | 76/84 | 76/84 |
| ตรวจพบจาก local อย่างเดียว | 74/84 | 74/84 |
| Mean segment IoU | 44.65% | 56.53% |
| Mean pixel recall | 47.99% | 73.23% |
| Mean pixel precision | 81.49% | 67.58% |
| Median segmentation time (synthetic) | 1.58 ms | 3.40 ms |

IoU คือสัดส่วนพื้นที่ทับซ้อนต่อพื้นที่รวมของ Segment กับ mask รอยจำลอง; recall บอกว่าครอบรอยได้มากแค่ไหน; precision บอกว่าพื้นที่ที่วงอยู่บนรอยจริงมากแค่ไหน ค่าเฉลี่ยรวมกรณีที่ไม่มี segment ด้วย

ผล PASS/FAIL ของ test ไม่มีภาพใดเปลี่ยน ผลดีขึ้นหลักๆ คือขอบเขต Segment แต่มีการวงเกินรอยมากขึ้น ค่าเวลาเฉพาะขั้น segmentation ไม่รวม PatchCore, จับภาพ, YOLO, marking หรือวาด Qt และเป็นการวัดหนึ่งรอบต่อภาพ

| รอยจำลอง | FAIL v1 / v2 | IoU v1 | IoU v2 |
|---|---:|---:|---:|
| Scratch | 14/21 ทั้งคู่ | 19.65% | 28.24% |
| Spot | 21/21 ทั้งคู่ | 65.53% | 69.95% |
| Chip | 21/21 ทั้งคู่ | 46.25% | 68.84% |
| Pin discoloration | 20/21 ทั้งคู่ | 47.15% | 59.10% |

## ข้อสรุปและข้อจำกัด

v2 เหมาะเป็น candidate สำหรับ Segment ที่ครอบบริเวณรอยมากขึ้น ยังไม่มีหลักฐานว่าตัดสิน PASS/FAIL แม่นขึ้น จึงคง v1 ในโปรแกรมจริงไว้ก่อน

รอยที่พลาด 8 ภาพคือ scratch 7 และ pin discoloration 1; raw peak ภายใน mask รอยอยู่ประมาณ 51.56–61.31 ซึ่งต่ำกว่า high threshold ที่เลือก ไม่ควรสรุปว่าโมเดลไม่เห็นทั้งหมด แต่สัญญาณยังไม่ผ่านเกณฑ์ candidate นี้

ของดี test เพียง 21 ภาพไม่เพียงพอยืนยันอัตรา false reject ใน production และ synthetic ไม่ใช่หลักฐานแทนของเสียจริง การจำลองสีขาเป็นการวางรอยในแถบขอบภาพ ไม่มีการยืนยันตำแหน่งขาจริงทุกภาพ ไม่มีกรณีทดสอบขาหาย

ควรตรวจภาพตัวอย่างและเพิ่มภาพจริงจากหลายสภาพแสง/ตำแหน่งก่อนเลือกใช้ v2 หากปรับเกณฑ์ตาม test รอบนี้ ต้องถือว่า test ชุดนี้กลายเป็นข้อมูลพัฒนา และใช้ชุดใหม่ยืนยันผลรอบถัดไป

## ไฟล์และการรันซ้ำ

- `candidate.json`: ค่า candidate ที่ตรึงก่อน test และข้อมูล calibration
- `comparison.json`: metrics, region statistics, model identity และ dataset audit
- `comparison.csv`: ผลรายภาพ พร้อม old/new verdict, threshold, peak, mean, area และ overlap
- `preview_01.png` ถึง `preview_12.png`: รอยจำลอง test 12 ภาพแรกตามลำดับคงที่ ซ้าย v1 ขวา v2 เส้นเหลืองคือ Segment เส้นขาวคือ mask รอยจำลอง

```powershell
python tools/compare_decision_v2.py --output reports/decision_v2_new_run
```

โฟลเดอร์ผลต้องยังไม่มีเพื่อป้องกันทับรายงานเก่า เครื่องมือไม่แก้ Recipe และไม่เปิดใช้ v2 อัตโนมัติ
