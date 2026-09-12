from ultralytics import YOLO

model = YOLO(r"C:\Users\nickk\Documents\cyn-x vision\models\best.pt")

results = model.predict(
    source=r"C:\Users\nickk\Documents\cyn-x vision\dataset\images\train\weed_pen_1.jpg",
    conf=0.01,
    save=True,
    verbose=True,
)

for result in results:
    print("Boxes:", len(result.boxes))

    for box in result.boxes:
        print(
            "Class:",
            int(box.cls[0]),
            "Confidence:",
            float(box.conf[0]),
        )