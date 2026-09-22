AI Tracking Turret
An AI‑powered tracking turret that uses YOLOv8 to detect people, lock onto the largest target, and track their chest position. The system includes sticky tracking, red‑hat filtering, and smooth reacquisition to keep the aim stable.
Runs in software‑only mode with on‑screen visualization. Hardware control can be added later.

Features
Real‑time person detection

Chest‑point targeting

Sticky target tracking

Automatic reacquisition

Red‑hat filtering

Armed / disarmed modes

On‑screen HUD and status display

Requirements
Python 3.8+

OpenCV

Ultralytics YOLOv8

A webcam

YOLO model file (yolov8n.pt)

Usage
Run the program and use the on‑screen display to track targets.

Controls:

A — Arm / Disarm

Q — Quit
