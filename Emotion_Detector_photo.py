import os
import sys
import cv2
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
import warnings
from torchvision.transforms.transforms import ToPILImage
#warnings.filterwarnings("ignore")

# Startup des Programms
print("############# Emotion Detector - V0.1 #############")
print("############# Sundermann / Teichert-Kluge #############")
print("Python version: ", sys.version)
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Laden des Face-Classifiers aus dem cv2 Paket
face_classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

# Laden der Klassen des Neural Networks
from WSCNet_Classes import *
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
model = torch.load(r'C:\Users\Nutzer\Documents\SeminarArbeit\Emotion_Detector\Emotion_Detector\wscnet_full_50runs_120622.pt',map_location=torch.device('cpu'))
model.eval()

# Disable Gradient
for param in model.parameters():
    param.grad = None


# Daten Preprocessing
data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(448),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


cam = cv2.VideoCapture(0)


def main():
    with torch.no_grad():
        cv2.namedWindow("test")
        img_counter = 0
        while True:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("test", frame)

            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                extracted_frame = cv2.flip(frame, 1)
                extracted_frame_gray = cv2.cvtColor(extracted_frame, cv2.COLOR_BGR2GRAY)
                faces = face_classifier.detectMultiScale(extracted_frame_gray, 1.3, 5) # Erkennen von Faces durch cv2 

                for (x, y, w, h) in faces:

                    cv2.rectangle(extracted_frame, (x, y), (x + w, y + h), (255, 255, 255), 2) # Erzeugen einer Box um Gesichter
                    ttens = data_transforms(extracted_frame[y : y + h, x : x + w]) # Transformieren der Bildauschnitte mit Gesichtern

                    if ttens is not None:

                        tensor, dec_tens = model(ttens[None]) # Prediction der Gesichter im WSCNet
                        pred = torch.max(tensor, dim=1)[1].tolist()
                        label = class_labels[pred[0]]
                        label_position = (x, y)
                        cv2.putText(extracted_frame, label, label_position, cv2.FONT_HERSHEY_DUPLEX, 1,  (255, 255, 255), 2)              
                    else:
                        cv2.putText(extracted_frame, "No face", (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

                cv2.imshow("Emotion Detector", extracted_frame)
                img_name = "open_cv_img/example_detection{}.png".format(img_counter)
                cv2.imwrite(img_name, extracted_frame)
                img_counter += 1
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
