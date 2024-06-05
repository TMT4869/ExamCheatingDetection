import cv2
import os

def capture_frame():
    # Mở webcam
    cap = cv2.VideoCapture(0)

    frame_count = 0
    count = 0
    while True:
        # Đọc frame từ webcam
        ret, frame = cap.read()

        # Hiển thị frame
        cv2.imshow('Webcam', frame)

        # Đếm số frame
        frame_count += 1

        # Lấy ảnh sau 20 frame
        if frame_count == 50:
            # Tạo thư mục nếu chưa tồn tại
            folder_path = r"E:\20222\Dataset\20193294"
            os.makedirs(folder_path, exist_ok=True)

            # Lưu ảnh vào thư mục với tên được đánh số tăng dần
            file_name = f"{count}.jpg"
            file_path = os.path.join(folder_path, file_name)
            cv2.imwrite(file_path, frame)
            print(f"Ảnh đã được lưu: {file_path}")

            # Tăng biến đếm để chuẩn bị lưu ảnh tiếp theo
            frame_count = 0
            count += 1

        # Thoát khỏi vòng lặp khi nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng webcam và đóng cửa sổ hiển thị
    cap.release()
    cv2.destroyAllWindows()

# Gọi hàm capture_frame để bắt đầu quá trình lấy ảnh từ webcam
capture_frame()
