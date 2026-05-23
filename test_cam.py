import cv2, time, sys

print(f"OpenCV: {cv2.__version__}")

for idx in [0, 1]:
    print(f"\n--- Trying index {idx} with CAP_AVFOUNDATION ---")
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    print(f"  isOpened: {cap.isOpened()}")
    if cap.isOpened():
        for i in range(15):
            ret, frame = cap.read()
            if ret:
                print(f"  ✅ Got frame on attempt {i+1}: {frame.shape}")
                break
            time.sleep(0.1)
        else:
            print("  ❌ No frame after 15 attempts")
        cap.release()
    else:
        print("  ❌ Could not open")

print("\nDone.")
