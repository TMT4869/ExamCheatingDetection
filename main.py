import csv
import cv2
import time
import gc
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import imutils
import numpy as np
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import torch
import pickle
from src.FaceAntiSpoofing import AntiSpoof

pw_mode = True
torch.set_num_threads(1)
username = ""
password = ""

# Load the face detection model
prototxt_path = "saved_models/deploy.prototxt.txt"
model_path = "saved_models/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Load the face recognition model
recognizer_path = "saved_models/recognizer.pkl"
anti_spoof = AntiSpoof('saved_models/AntiSpoofing_print-replay_1.5_128.onnx')
svm_model = pickle.load(open(recognizer_path, 'rb'))
resnet = InceptionResnetV1(pretrained='vggface2').eval()

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class FirstPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        def check_user():
            with open("database.csv", mode="r") as f:
                reader = csv.reader(f, delimiter=",")
                for row in reader:
                    if row[0]== username and row[1] == password:
                        check = True
                        break
                    else:
                        check = False
            return check

        def signin():
            global username
            global password
            username = user.get()
            password = pw.get()
            if (username == "" or username == "Username") or (password == "" or password == "Password"):
                messagebox.showerror("Entry error", "Type username or password")
            else:
                if (check_user() == True):
                    controller.show_frame(SecondPage)
                else:
                    messagebox.showinfo("Invalid", "Invalid username or password")

        img = tk.PhotoImage(file="Login.png")
        label = tk.Label(self, image=img, bg="white")
        label.image = img
        label.place(x=80, y=60)

        frame = tk.Frame(self, width=350, height=350, bg="white")
        frame.place(x=510, y=80)

        heading = tk.Label(frame, text="Sign in", fg="#57a1f8", bg="white", font=("Microsoft YaHei UI Light", 23, "bold"))
        heading.place(x=150, y=15)

        def user_enter(event):
            user.delete(0, "end")

        def user_leave(event):
            name = user.get()
            if name == "":
                user.insert(0, "Username")

        user = tk.Entry(frame, width=25, fg="black", border=0, bg="white", font=("Microsoft YaHei UI Light", 11))
        user.place(x=60, y=90)
        user.insert(0,"Username")
        user.bind("<FocusIn>", user_enter)
        user.bind("<FocusOut>", user_leave)
        tk.Frame(frame, width=295, height=2, bg="black").place(x=55, y=117)


        def pw_enter(event):
            pw.delete(0, "end")

        def pw_leave(event):
            name = pw.get()
            if name == "":
                pw.insert(0, "Password")

        pw = tk.Entry(frame, width=25, fg="black", border=0, bg="white", font=("Microsoft YaHei UI Light", 11))
        pw.place(x=60, y=160)
        pw.insert(0, "Password")
        pw.bind("<FocusIn>", pw_enter)
        pw.bind("<FocusOut>", pw_leave)
        tk.Frame(frame, width=295, height=2, bg="black").place(x=55, y=187)


        def hide():
            global pw_mode
            if pw_mode:
                eye_button.config(image=close_eye, activebackground="white")
                pw.config(show='*')
                pw_mode = False
            else:
                eye_button.config(image=open_eye, activebackground="white")
                pw.config(show='')
                pw_mode = True

        open_eye = tk.PhotoImage(file="openeye.png")
        close_eye = tk.PhotoImage(file="closeeye.png")
        eye_button = tk.Button(frame, image=open_eye, bg="white", bd=0, command=hide)
        eye_button.place(x=330, y=160)

        tk.Button(frame, width=39, pady=7, text="Sign in", bg="#57a1f8", fg="white", border=0, command=signin).place(x=65, y=214)
        label = tk.Label(frame, text="Don't have an account?", fg="black", bg="white", font=("Microsoft YaHei UI Light", 9))
        label.place(x=105, y=280)

        sign_up = tk.Button(frame, width=6, text="Sign up", border=0, bg="white", cursor="hand2", fg="#57a1f8")
        sign_up.place(x=245, y=280)

class SecondPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.is_running = False

        self.cheat_detected_time = time.time()
        self.is_cheat = False
        self.face_count = 0
        self.threshold = 3

        self.webcam_label = tk.Label(self)
        self.webcam_label.place(x=20, y=15, width=640, height=480)

        self.username_label = tk.Label(self, text=f"ID Student: {username}", bg="white", font=("Microsoft YaHei UI", 11),
                                       width=25, anchor="center")
        self.username_label.place(x=680, y=50)

        btn_logout = tk.Button(self, text="Log out", bg="#57a1f8", fg="white", font=("Microsoft YaHei UI", 10),\
                           width=20, pady=8, border=0, command=self.logout)
        btn_logout.place(x=715, y=450)

        btn_start = tk.Button(self, text="Start", bg="#57a1f8", fg="white", font=("Microsoft YaHei UI", 10), \
                           width=20, pady=8, border=0, command=self.face_detection)
        btn_start.place(x=715, y=380)
        self.btn_start = btn_start

    def logout(self):
        self.is_running = False
        self.webcam_label.config(image='')
        self.btn_start.config(state="normal")
        self.controller.show_frame(FirstPage)

    def face_detection(self):
        self.is_running = True
        self.btn_start.config(state="disabled")
        self.username_label.config(text=f"ID Student: {username}")

        def increased_crop(img, bbox: tuple, bbox_inc: float = 1.5):
            # Crop face based on its bounding box
            real_h, real_w = img.shape[:2]

            x, y, w, h = bbox
            w, h = w - x, h - y
            l = max(w, h)

            xc, yc = x + w / 2, y + h / 2
            x, y = int(xc - l * bbox_inc / 2), int(yc - l * bbox_inc / 2)
            x1 = 0 if x < 0 else x
            y1 = 0 if y < 0 else y
            x2 = real_w if x + l * bbox_inc > real_w else x + int(l * bbox_inc)
            y2 = real_h if y + l * bbox_inc > real_h else y + int(l * bbox_inc)

            img = img[y1:y2, x1:x2, :]
            img = cv2.copyMakeBorder(img,
                                     y1 - y, int(l * bbox_inc - y2 + y),
                                     x1 - x, int(l * bbox_inc) - x2 + x,
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
            return img

        def process_frame(frame):
            gc.collect()
            # Resize frame for faster processing
            frame = imutils.resize(frame, width=640)
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            face_count = np.sum(detections[0][0][:, 2] > 0.7)
            if face_count == 0:
                self.is_cheat = False

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.7:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # extract the face ROI
                    if (startX >= 0) and (startY >= 0) and (endX >= 0) and (endY >= 0):
                        face = frame[startY:endY, startX:endX]
                        face = Image.fromarray(face)

                        # Perform anti-spoofing check
                        pred = anti_spoof([increased_crop(frame, (startX, startY, endX, endY), bbox_inc=1.9)])[0]
                        label = np.argmax(pred)
                        real_face_score = pred[0][0]

                        if label == 0:
                            # Face is real, perform recognition
                            # perform classification to recognize the face
                            face = transform(face).unsqueeze(0)
                            with torch.no_grad():
                                embeddings = resnet(face)
                            face_embeddings = embeddings.flatten().reshape(1, -1)
                            name = svm_model.predict(face_embeddings)[0]

                            if face_count <= 1 and name == username:
                                self.is_cheat = False
                                color = (0, 255, 0)
                            else:
                                self.is_cheat = True
                                color = (255, 0, 0)

                            # draw the bounding box of the face along with the
                            # associated probability
                            text = "{}: {:.2f}%".format(name, real_face_score * 100)
                            y = startY - 10 if startY - 10 > 10 else startY + 10
                            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                        else:
                            text = "Fake face"
                            y = startY - 10 if startY - 10 > 10 else startY + 10
                            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
                            self.is_cheat = True

            return frame

        def show_frame():
            if not self.is_running:
                cap.release()
                return
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = process_frame(frame)

                if self.is_cheat:
                    current_time = time.time()
                    if current_time - self.cheat_detected_time >= 3:
                        messagebox.showwarning("Warning", "Cheat!!!")
                        self.cheat_detected_time = current_time
                else:
                    self.cheat_detected_time = time.time()

                image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image)

                self.webcam_label.configure(image=photo)
                self.webcam_label.image = photo

                self.webcam_label.after(70, show_frame)
            else:
                cap.release()

        # Open the video capture
        cap = cv2.VideoCapture(0)

        # Start displaying the video stream
        show_frame()

class Application(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.geometry("925x520+300+200")
        self.configure(bg="white")
        self.resizable(False, False)

        # creating a window
        window = tk.Frame(self)
        window.pack()

        window.grid_rowconfigure(0, minsize=520)
        window.grid_columnconfigure(0, minsize=925)

        self.frames = {}
        for F in (FirstPage, SecondPage):
            frame = F(window, self)
            frame.configure(bg="white")
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(FirstPage)

    def show_frame(self, page):
        frame = self.frames[page]
        frame.tkraise()
        self.title("Application")

        if page == SecondPage:
            frame.username_label.config(text=f"ID Student: {username}")


if __name__ == "__main__":
    app = Application()
    app.mainloop()