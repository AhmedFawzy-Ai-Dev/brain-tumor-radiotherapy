# LinkedIn post — draft

Attach, in this order: `docs/images/demo.gif`, `docs/images/islands_before_after.png`,
`docs/images/volume_agreement.png`, then `docs/images/report_pdf.png`.

Numbers come from `docs/eval/*.json` (30 held-out patients); keep them in sync
if the eval is re-run.

---

## English

My brain-tumour report said a tumour's longest diameter was 107 mm. The line drawn on the picture right next to it was 53 mm long. 🧠📏

The segmentation was fine; my measurement code wasn't. It took the two farthest points of the whole mask, and the expert's own mask had a 0.14 cm³ speck far from the tumour. Dice can't catch that: Dice doesn't measure anything in millimetres.

So I rebuilt the project around the numbers a clinician actually reads:

📥 Pulled 120 BraTS patients out of a 7.6 GB archive by downloading only the byte ranges I needed (parallel + resumable): 1.3 GB instead of 7.6
🔒 Fixed a train / val / test split: 30 test patients the new model never saw, and none the first model saw either
📏 Scored whole scans, per patient: Dice, boundary error in mm (HD95), and how far the report's volume and diameters are from the same measurements on the expert's mask

What I found:

🐞 On the 30 test patients' expert masks, specks under 1 cm³ (down to a single voxel) stretched a reported diameter in 9 patients, by up to 26 mm. Diameters are now measured on the lesion itself, and a genuine second region is flagged in the report instead of being measured across.

📊 Then, same recipe, 24 → 80 training patients:
• Report's tumour volume error: 19.5% → 9.1% (median), within 10% of the expert for 16 of 30 patients instead of 8
• Diameter errors: 9–10 mm → ~5.5 mm
• Whole-tumour Dice: 0.78 → 0.86

Two things surprised me:
🔹 Flip test-time augmentation, a standard trick, didn't pay for itself: +0.002 Dice at twice the compute, and a slightly worse volume error. It's off by default.
🔹 The model still under-measures (−10 cm³ on average). More data halved that bias but didn't remove it, and cleaning up the prediction differently explains only about a third of it.

Also in the box:
✅ Bilingual PDF report (English + Arabic). I fixed Arabic lines coming out in reverse order: the classic "reorder, then wrap" bug
✅ Web UI: upload a scan, get the slice with the measured line, the measurements and the PDF, in ~20 s on a laptop CPU
✅ Tumour type on 2-D clinical images: 96.1% accuracy on 1,311 test images
✅ 36 tests, CI, and every number reproducible with `make`

The honest part: it's a 2-D model trained on a laptop CPU with 80 patients; 3-D models trained on the full dataset with GPUs do better. It's a research prototype, not a medical device.

💻 Code, per-patient results and charts: https://github.com/AhmedFawzy-Ai-Dev/brain-tumor-radiotherapy

#MedicalImaging #DeepLearning #ComputerVision #PyTorch #MachineLearning #AI #Radiotherapy

---

## العربي

التقرير اللي بيطلعه مشروعي قال إن أطول قطر للورم 107 مم… والخط المرسوم على الصورة اللي جنبه على طول كان طوله 53 مم. 🧠📏

الـ segmentation كانت سليمة، الغلط كان في كود القياس: كان بياخد أبعد نقطتين في الـ mask كلها، وماسك الخبير نفسه كان فيها نقطة صغيرة جداً (0.14 سم³) بعيدة عن الورم. والـ Dice مستحيل يكشف حاجة زي دي، لأن الـ Dice مابيقيسش أي حاجة بالملّيمتر.

فبنيت المشروع من الأول حوالين الأرقام اللي الدكتور بيقراها فعلاً:

📥 نزّلت 120 مريض من BraTS من أرشيف حجمه 7.6 جيجا، بتحميل الأجزاء اللي محتاجها بس (بالتوازي ومع استكمال لو النت قطع): 1.3 جيجا بدل 7.6
🔒 ثبّت تقسيم train / val / test: فيه 30 مريض للاختبار الموديل الجديد ماشافهمش، ولا الموديل الأول كمان
📏 قيّمت على الـ scan كاملة لكل مريض: الـ Dice، وخطأ الحدود بالمليمتر (HD95)، والأهم: الحجم والأقطار اللي في التقرير بعيدة قد إيه عن نفس القياسات على ماسك الخبير

اللي لقيته:

🐞 على ماسكات الخبراء للـ 30 مريض: نقط أصغر من 1 سم³ (أحياناً voxel واحد) كبّرت القطر في 9 مرضى، لحد 26 مم. دلوقتي القطر بيتقاس على الورم نفسه، ولو فيه منطقة تانية حقيقية التقرير بيقول كده بدل ما يقيس عبرها.

📊 وبعدين بنفس الطريقة بالظبط، من 24 لـ 80 مريض في التدريب:
• خطأ حجم الورم في التقرير: من 19.5٪ لـ 9.1٪، وبقى في حدود 10٪ من الخبير لـ 16 مريض من 30 بدل 8
• خطأ الأقطار: من 9–10 مم لحوالي 5.5 مم
• الـ Dice للورم كله: من 0.78 لـ 0.86

حاجتين فاجأوني:
🔹 الـ flip test-time augmentation، وهي حيلة مشهورة، ماكانتش تستاهل: 0.002+ في الـ Dice بضعف وقت الحساب، وخطأ الحجم بقى أسوأ شوية. عشان كده مقفولة افتراضياً.
🔹 الموديل لسه بيقلّل حجم الورم (بمتوسط 10 سم³). الداتا الأكتر قسمت الانحياز ده على اتنين بس ماشالتهوش، وتغيير المعالجة اللي بعد التوقّع بيفسّر حوالي تلته بس.

وكمان:
✅ تقرير PDF بالعربي والإنجليزي: صلّحت إن السطور العربي كانت بتطلع بالمقلوب (أول الجملة في آخر سطر)
✅ واجهة ويب: ترفع الـ scan وتاخد الشريحة عليها الخط اللي اتقاس، والقياسات، والـ PDF، في حوالي 20 ثانية على CPU لابتوب
✅ تصنيف نوع الورم على صور 2-D: دقة 96.1٪ على 1,311 صورة اختبار
✅ 36 test و CI، وكل رقم ممكن يتعاد بأمر `make`

وبصراحة: ده موديل 2-D متدرّب على CPU لابتوب بـ 80 مريض، والموديلات الـ 3-D المتدرّبة على الداتا كاملة بـ GPU أحسن منه. وده نموذج بحثي، مش جهاز طبي.

💻 الكود ونتايج كل مريض والرسوم: https://github.com/AhmedFawzy-Ai-Dev/brain-tumor-radiotherapy

#MedicalImaging #DeepLearning #AI #PyTorch #MachineLearning
