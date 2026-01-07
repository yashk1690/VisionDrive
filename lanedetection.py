import cv2
import numpy as np


def convert_video_to_grayscale(input_path, output_path):
    # 1. Create a VideoCapture object
    cap = cv2.VideoCapture(input_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 2. Get video properties to configure the VideoWriter
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    # 'mp4v' is a common codec for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Note: isColor=False is crucial for writing grayscale correctly
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    print("Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # 3. Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 4. Write the grayscale frame to the output video
            out.write(gray_frame)

            # Optional: Display the frame while processing (press 'q' to quit)
            cv2.imshow('Frame', gray_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # 5. Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done! Saved to {output_path}")


# --- Usage ---
input_video = 'my_video.mp4'  # Replace with your input file name
output_video = 'output_grey.mp4'  # Replace with your desired output name

convert_video_to_grayscale(input_video, output_video)