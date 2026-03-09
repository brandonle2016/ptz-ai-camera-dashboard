import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=1920,
    display_height=1080,
    framerate=30,
    flip_method=2,
):
    """
    Constructs the GStreamer pipeline string for OpenCV.
    Notice the end converts to BGR, which is what OpenCV and YOLO expect.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )

def test_camera():
    # Generate the pipeline string
    model = YOLO("yolo26n.engine")
    pipeline = gstreamer_pipeline(flip_method=2)
    print(f"Using pipeline: \n{pipeline}")

    # Open the camera feed using the GStreamer backend
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera opened successfully. Press 'q' to quit.")

    try:
        while True:
            # Read a frame from the pipeline
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break
            else:
            	results=model.track(frame,persist=True)
            	result_frame=results[0].plot()
            # --- THIS IS WHERE YOUR YOLO INFERENCE WILL GO ---
            # results = model(frame)
            # frame = results.render()[0]

            # Display the frame
            #cv2.imshow("Orin Nano CSI Camera", frame)

            # Break the loop if 'q' is pressed
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
    finally:
        # Clean up
        cap.release()
        #cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()
