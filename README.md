# task2-on-AI
this is a full description of the color recognition project

# Color Recognition using OpenCV

## Project Overview
This project is a simple Color Recognition application built using Python and OpenCV.

The program captures live video from the webcam, converts each frame from the BGR color space to the HSV color space, and detects specific colors in real time. When a colored object is detected, the program draws a bounding box around it and displays the color name.

The project currently recognizes the following colors:
- Red
- Green
- Blue
- Yellow
- Orange
- Purple

---

# Features

- Real-time webcam capture
- Detect multiple colors simultaneously
- Draw bounding boxes around detected objects
- Display the detected color name
- Count the number of detected objects for each color
- Display FPS (Frames Per Second)
- Save screenshots
- Record video
- Show binary mask
- Calibration mode for creating new colors
- Save custom colors in a JSON configuration file

---

# Requirements

- Python 3.x
- OpenCV
- NumPy

Install the required libraries:

```bash
pip install opencv-python numpy
```

---

# How the Program Works

## Step 1: Import Libraries

The program imports the required libraries.

- OpenCV (cv2) for image processing.
- NumPy for handling arrays.
- Time for FPS calculation.
- JSON for saving color configurations.
- OS for checking files.
- Deque for storing movement history.

---

## Step 2: Load Color Configuration

The program loads the default HSV color ranges.

If a configuration file (`color_ranges.json`) already exists, it loads the saved colors instead.

---

## Step 3: Open the Camera

```python
cap = cv2.VideoCapture(0)
```

The webcam starts capturing live video.

---

## Step 4: Capture Video Frames

```python
ret, frame = cap.read()
```

- `ret` indicates whether the frame was captured successfully.
- `frame` contains the current image from the webcam.

The program continuously reads frames inside a loop to create live video.

---

## Step 5: Convert BGR to HSV

```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
```

The webcam captures images in the BGR color space.

HSV makes color detection easier because it separates:

- Hue (Color)
- Saturation (Intensity)
- Value (Brightness)

---

## Step 6: Create Color Masks

Each color has its own HSV range.

The program creates a binary mask using:

```python
cv2.inRange()
```

Pixels inside the selected color range become white.

All other pixels become black.

---

## Step 7: Remove Noise

The mask is cleaned using:

- Erosion
- Dilation

These operations remove small unwanted pixels and improve detection accuracy.

---

## Step 8: Detect Objects

The program finds contours using:

```python
cv2.findContours()
```

Each contour represents one detected object.

---

## Step 9: Filter Small Objects

The contour area is calculated.

Objects smaller than the minimum area are ignored to reduce false detections.

---

## Step 10: Draw Bounding Boxes

For every detected object, the program:

- Draws a rectangle.
- Calculates the object's center.
- Displays the color name.
- Shows the detected area.

---

## Step 11: Draw Motion Trail

The center of each detected object is stored.

As the object moves, the program draws a trail showing its previous positions.

---

## Step 12: Calculate FPS

The program calculates Frames Per Second (FPS) to measure processing performance.

---

## Step 13: Display Information

The interface displays:

- FPS
- Minimum detection area
- Number of detected objects for each color
- Recording status

---

## Step 14: Keyboard Controls

| Key | Function |
|------|----------|
| Q | Quit the program |
| C | Enable calibration mode |
| A | Save a new color |
| M | Show/Hide mask |
| S | Save screenshot |
| R | Start/Stop video recording |
| + | Increase minimum object area |
| - | Decrease minimum object area |

---

# Technologies Used

- Python
- OpenCV
- NumPy

---

# Future Improvements

- Detect more colors automatically.
- Improve object tracking accuracy.
- Add object recognition using AI models.
- Support external cameras.
- Create a graphical user interface (GUI).

---
