# 🚘 Lane Detection System using OpenCV

A robust, real-time lane detection pipeline built with Python and OpenCV. This script is designed for Advanced Driver Assistance Systems (ADAS) and processes road images to detect and visualize lane markings using image processing techniques and Hough Transforms.

> **This project is fully open source and free to use, modify, and distribute under the MIT License.** Whether you're a student, researcher, or developer, you're welcome to build upon this work.

## 📌 Features

- ✅ Adaptive color filtering for detecting white and yellow lanes  
- ✅ CLAHE-based contrast enhancement for better visibility  
- ✅ Uses both Standard and Probabilistic Hough Transforms for robust line detection  
- ✅ Real-time visualization of processing pipeline  
- ✅ Automatic steering direction suggestion (Left, Right, Centered)  
- ✅ Displays histogram of edge pixel intensities  
- ✅ Accuracy measurement of lane detection across frames

## 🛠 Technologies Used

- **Language:** Python 3  
- **Libraries:**  
  - OpenCV  
  - NumPy  
  - Matplotlib

## 🗂 Dataset Requirement

Place your dataset in a folder structured like this:

/your-dataset-folder/  
├── frame1.jpg  
├── frame2.jpg  
├── ...

Update the following line in the script to point to your dataset:

image_folder = '/path/to/your/image/folder'

Supported image formats: .jpg, .jpeg, .png

## 📦 Installation

1. Clone this repository:

git clone https://github.com/yourusername/lane-detection-opencv.git  
cd lane-detection-opencv

2. Install dependencies:

pip install opencv-python numpy matplotlib

## ▶️ How to Run

1. Make sure your dataset folder is configured in the code.  
2. Run the script:

python lane_detection.py

3. You'll see a window displaying the full processing pipeline:  
   - Original frame  
   - Enhanced contrast  
   - Color-filtered result  
   - Region of interest (ROI)  
   - Grayscale and blurred image  
   - Canny edge detection result  
   - Final output with annotated lanes and driving direction  

4. A real-time histogram will also be shown for lane center estimation.  
5. Press `q` to exit the visualization.

## 📊 Output

At the end of execution, you'll see a summary like:

Total Frames: 200  
Correct Lane Detections (both lanes): 178  
Lane Detection Accuracy: 89.00%

## ⚙️ Customization Tips

- Change contrast settings in enhance_contrast()  
- Modify HLS color filter ranges in color_filter()  
- Adjust thresholds for edge and line detection  
- Tune MAX_HISTORY to smooth lane predictions

## 🧪 Pipeline Stages

1. Enhance image contrast  
2. Color mask for yellow & white lanes  
3. ROI cropping  
4. Grayscale + Gaussian blur  
5. Canny edge detection  
6. Standard & Probabilistic Hough Transform  
7. Lane averaging and tracking  
8. Direction estimation  
9. Visualization and accuracy reporting

## 📄 License

**MIT License**

This project is licensed under the MIT License — meaning it's completely open source. You're free to:  
- ✅ Use it for personal or commercial projects  
- ✅ Modify and adapt it to your own needs  
- ✅ Share or distribute your own versions  

Please consider keeping a reference to the original repository.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files...

(Include the full license text in a LICENSE file.)

## 👨‍💻 Author

Developed by [Mahmoud Fouad, Omar Nagy, Habiba Tamer]  
Feel free to fork, contribute, or share! If this helps you, consider starring the repo.



