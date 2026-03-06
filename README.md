# AI Steel Surface Defect Monitoring System

An industrial computer vision system for automated quality inspection in smart manufacturing.

This project detects surface defects in steel sheets using a YOLOv8 model and provides real-time analytics through a FastAPI backend and Streamlit dashboard.

---

## Features

• YOLOv8 defect detection  
• ONNX optimized inference  
• FastAPI production API  
• Streamlit monitoring dashboard  
• Defect severity classification  
• Manufacturing analytics  

---

## System Architecture

Image Upload / Camera
        ↓
Streamlit Dashboard
        ↓
FastAPI Inference API
        ↓
YOLOv8 ONNX Model
        ↓
Defect Detection + NMS
        ↓
Severity Analysis
        ↓
Production Analytics

---

## Installation

Install dependencies

pip install -r requirements.txt

Start API

uvicorn app.main:app --reload

Start dashboard

streamlit run dashboard.py

---

## Demo

Example detection and analytics dashboard screenshots are available in the demo folder.
