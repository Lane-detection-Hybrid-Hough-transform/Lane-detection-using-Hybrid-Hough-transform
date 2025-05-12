import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Constants for line averaging
lane_history = {"left": [], "right": []}
MAX_HISTORY = 5

total_frames = 0
correct_detections = 0

def add_title(img, title):
    return cv2.putText(img.copy(), title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

def convert_color(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

def color_filter(img):
    hls = convert_color(img)
    lower_white = np.array([0, 120, 0])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(hls, lower_white, upper_white)

    lower_yellow = np.array([15, 30, 115])
    upper_yellow = np.array([35, 204, 255])
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

    mask = cv2.bitwise_or(white_mask, yellow_mask)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def average_lines(history, new_line, side):
    history[side].append(new_line)
    if len(history[side]) > MAX_HISTORY:
        history[side].pop(0)
    return np.mean(history[side], axis=0).astype(np.int32)

def detect_lane(frame):
    enhanced_frame = enhance_contrast(frame)
    color_filtered_frame = color_filter(frame)

    height, width = frame.shape[:2]
    roi = frame[int(0.45 * height):, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    median_intensity = np.median(blurred)
    lower_threshold = int(max(0, 0.7 * median_intensity))
    upper_threshold = int(min(255, 1.3 * median_intensity))
    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)

    lines_sht = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    lines_pht = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=40)

    left_lane_lines = []
    right_lane_lines = []

    if lines_sht is not None:
        for line in lines_sht:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if line_length < 3:
                    continue

                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
                if abs(slope) < 0.3 or abs(slope) > 6:
                    continue

                mid_x = width // 2
                if slope < -0.3 and x1 < mid_x and x2 < mid_x:
                    left_lane_lines.append([x1, y1, x2, y2])
                elif slope > 0.3 and x1 > mid_x and x2 > mid_x:
                    if x1 < width - 100 and x2 < width - 10:
                        right_lane_lines.append([x1, y1, x2, y2])

    if lines_pht is not None:
        for line in lines_pht:
            for x1, y1, x2, y2 in line:
                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if line_length < 3:
                    continue

                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
                if abs(slope) < 0.2 or abs(slope) > 10:
                    continue

                mid_x = width // 2
                if slope < 0 and x1 < mid_x + 50 and x2 < mid_x + 50 and x1 > 10 and x2 > 10:
                    left_lane_lines.append([x1, y1, x2, y2])
                elif 4 > slope > 0.3 and x1 > mid_x - 20 and x2 > mid_x - 20:
                    right_lane_lines.append([x1, y1, x2, y2])

    left_detected = False
    right_detected = False
    annotated_frame = frame.copy()

    if left_lane_lines:
        left_detected = True
        left_lane_avg = np.mean(left_lane_lines, axis=0, dtype=np.int32)
        left_lane_avg = average_lines(lane_history, left_lane_avg, "left")
        x1, y1, x2, y2 = left_lane_avg
        roi_offset = int(0.5 * height)
        cv2.line(annotated_frame, (x1, y1 + roi_offset), (x2, y2 + roi_offset), (0, 255, 0), 3)

    if right_lane_lines:
        right_detected = True
        right_lane_avg = np.mean(right_lane_lines, axis=0, dtype=np.int32)
        right_lane_avg = average_lines(lane_history, right_lane_avg, "right")
        x1, y1, x2, y2 = right_lane_avg
        roi_offset = int(0.5 * height)
        cv2.line(annotated_frame, (x1, y1 + roi_offset), (x2, y2 + roi_offset), (0, 255, 0), 3)

    if left_detected and right_detected:
        y_bottom = height - int(height / 4)

        def get_line_eq(x1, y1, x2, y2):
            if x2 == x1:
                return float('inf'), x1
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            return m, b

        m_left, b_left = get_line_eq(*left_lane_avg)
        m_right, b_right = get_line_eq(*right_lane_avg)

        x_left = int((y_bottom - b_left) / m_left) if m_left != float('inf') else b_left
        x_right = int((y_bottom - b_right) / m_right) if m_right != float('inf') else b_right

        lane_center = (x_left + x_right) // 2
        frame_center = width // 2

        offset = lane_center - frame_center

        hist_region = edges[-100:, :]
        hist = np.sum(hist_region, axis=0)
        left_peak = np.argmax(hist[:width // 2])
        right_peak = np.argmax(hist[width // 2:]) + width // 2
        hist_center = (left_peak + right_peak) // 2
        hist_offset = hist_center - frame_center

        combined_offset = int((offset + hist_offset) / 2)

        if offset > 250:
            direction = "Turn Right"
        elif offset < -20:
            direction = "Turn Left"
        else:
            direction = "Stay Centered"

        cv2.putText(annotated_frame, direction, (width // 2 - 100, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        histogram = np.sum(edges[edges.shape[0] // 2:, :], axis=0)
        plt.clf()
        plt.title("Lane Position Histogram")
        plt.xlabel("Column Index")
        plt.ylabel("Pixel Intensity Sum")
        plt.plot(histogram)
        plt.axvline(x=frame_center, color='r', linestyle='--', label='Frame Center')
        plt.axvline(x=lane_center, color='g', linestyle='--', label='Lane Center')
        plt.legend()
        plt.pause(0.001)

    return annotated_frame, frame.copy(), enhanced_frame, color_filtered_frame, roi, gray, blurred, edges, left_detected, right_detected

# Main processing
image_folder = '/home/raspberry/Desktop/Advanced-Driver-Assistance-System-main/ADAS/TuSimple/0530'
image_files = []
for root, dirs, files in os.walk(image_folder):
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(root, f))
image_files = sorted(image_files)

if not image_files:
    print("Error: No image files found in the dataset.")
    exit()

for image_file in image_files:
    frame = cv2.imread(image_file)
    if frame is None:
        print(f"Failed to load {image_file}")
        continue

    results = detect_lane(frame)
    annotated_frame, original_frame, enhanced_frame, color_filtered_frame, roi, gray, blurred, edges, left_detected, right_detected = results

    total_frames += 1
    if left_detected and right_detected:
        correct_detections += 1

    target_size = (600, 450)

    original_view = add_title(cv2.resize(original_frame, target_size), "Original")
    enhanced_view = add_title(cv2.resize(enhanced_frame, target_size), "Enhanced")
    filtered_view = add_title(cv2.resize(color_filtered_frame, target_size), "Color Filter")
    roi_view = add_title(cv2.resize(roi, target_size), "ROI")

    gray_view = add_title(cv2.cvtColor(cv2.resize(gray, target_size), cv2.COLOR_GRAY2BGR), "Gray")
    blurred_view = add_title(cv2.cvtColor(cv2.resize(blurred, target_size), cv2.COLOR_GRAY2BGR), "Blurred")
    edges_view = add_title(cv2.cvtColor(cv2.resize(edges, target_size), cv2.COLOR_GRAY2BGR), "Canny")
    final_view = add_title(cv2.resize(annotated_frame, target_size), "Final Output")

    top_row = np.hstack((original_view, enhanced_view, filtered_view, roi_view))
    bottom_row = np.hstack((gray_view, blurred_view, edges_view, final_view))
    combined_display = np.vstack((top_row, bottom_row))

    cv2.namedWindow('Lane Detection Pipeline', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Lane Detection Pipeline', 800, 720)
    cv2.imshow('Lane Detection Pipeline', combined_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

accuracy = (correct_detections / total_frames) * 100 if total_frames > 0 else 0
print(f"\nTotal Frames: {total_frames}")
print(f"Correct Lane Detections (both lanes): {correct_detections}")
print(f"Lane Detection Accuracy: {accuracy:.2f}%")

cv2.destroyAllWindows()
