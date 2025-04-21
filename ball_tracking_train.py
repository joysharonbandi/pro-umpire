from ultralytics import YOLO
import tensorflow as tf
import os

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)


model_path = os.path.join('runs','detect','train5','weights','best.pt')
model = YOLO(model_path)
# model = YOLO("yolov8s.pt")
results = model.train(data="data.yaml", epochs=100)