import torch
import gc
import pickle
import cv2
import imutils
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from src.FaceAntiSpoofing import AntiSpoof
from torchvision import transforms

torch.set_num_threads(1)
args = {
  "input": "E://20222//Dataset//me_2.mp4",
  "output": "output.avi",
  "confidence": 0.7
}

# Load the face detection model
prototxt_path = "saved_models/deploy.prototxt.txt"
model_path = "saved_models/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
anti_spoof = AntiSpoof('saved_models/AntiSpoofing_print-replay_128.onnx')

# Load the face recognition model
embeddings_path = "saved_models/embeddings.pkl"
recognizer_path = "saved_models/recognizer.pkl"
labels_path = "saved_models/labels.pkl"

svm_model = pickle.load(open(recognizer_path, 'rb'))
labels = pickle.load(open(labels_path, 'rb'))

resnet = InceptionResnetV1(pretrained='vggface2').eval()

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def increased_crop(img, bbox : tuple, bbox_inc : float = 1.5):
    # Crop face based on its bounding box
    real_h, real_w = img.shape[:2]

    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)

    xc, yc = x + w/2, y + h/2
    x, y = int(xc - l*bbox_inc/2), int(yc - l*bbox_inc/2)
    x1 = 0 if x < 0 else x
    y1 = 0 if y < 0 else y
    x2 = real_w if x + l*bbox_inc > real_w else x + int(l*bbox_inc)
    y2 = real_h if y + l*bbox_inc > real_h else y + int(l*bbox_inc)

    img = img[y1:y2,x1:x2,:]
    img = cv2.copyMakeBorder(img,
                                y1-y, int(l*bbox_inc-y2+y),
                                x1-x, int(l*bbox_inc)-x2+x,
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

# grab a reference to the video file and initialize pointer to output
# video file
print("[INFO] opening video file...")
vs = cv2.VideoCapture(0)

# loop over frames from the video file stream
while True:
    gc.collect()
    # grab the current frame
    frame = vs.read()[1]

    # check to see if we have reached the end of the stream
    if frame is None:
        break

    # Resize frame for faster processing
    frame = imutils.resize(frame, width=640)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = img[startY:endY, startX:endX]
            face = Image.fromarray(face)

            # Perform anti-spoofing check
            pred = anti_spoof([increased_crop(img, (startX, startY, endX, endY), bbox_inc=2)])[0]
            label = np.argmax(pred)
            real_face_score = pred[0][0]

            # if label == 0:
                # Face is real, perform recognition
                # perform classification to recognize the face
            face = transform(face).unsqueeze(0)
            with torch.no_grad():
                embeddings = resnet(face)
            face_embeddings = embeddings.flatten().reshape(1, -1)
            name = svm_model.predict(face_embeddings)[0]

            # draw the bounding box of the face along with the
            # associated probability
            text = "{}".format(name)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                # text = "Real face"
                # y = startY - 10 if startY - 10 > 10 else startY + 10
                # cv2.rectangle(frame, (startX, startY), (endX, endY),
                #               (0, 255, 0), 2)
                # cv2.putText(frame, text, (startX, y),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            # else:
            #     text = "Fake face"
            #     y = startY - 10 if startY - 10 > 10 else startY + 10
            #     cv2.rectangle(frame, (startX, startY), (endX, endY),
            #                   (0, 0, 255), 2)
            #     cv2.putText(frame, text, (startX, y),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# do a bit of cleanup
vs.release()
# Destroy all the windows
cv2.destroyAllWindows()
