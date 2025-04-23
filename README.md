# 🌿 Plant Disease Detection

A web application that uses machine learning to detect plant diseases from images. This project features a Python backend and a modern frontend for an interactive user experience.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [License](#license)

## Features

- **Automated Detection:** Uses machine learning to identify plant diseases.
- **Interactive UI:** Seamless integration between backend predictions and frontend display.
- **Scalable Architecture:** Separates backend and frontend for easier maintenance and scaling.

## Prerequisites

- Python 3.8+
- Node.js and npm
- Basic understanding of command line operations

## Setup Instructions

### Backend Setup - Create and Activate Virtual Environment

1. **Clone the repository in your System:**

   ```sh
   git clone https://github.com/skabdulM/Plant-disease-detection.git
   ```

2. **Navigate to the project directory:**

   ```sh
   cd ./Plant-disease-detection
   ```

3. **Create a virtual environment:**

   ```sh
   python -m venv .venv
   ```

4. **Activate the virtual environment:**

   - On Windows:
     ```sh
     .\.venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```sh
     source .venv/bin/activate
     ```

5. **Start Server:**

   ```sh
   cd ../
   python app.py
   ```

### Frontend Setup

1. **Install the dependencies:**

   ```sh
   cd frontend
   npm install
   ```

2. **Start Server:**

   ```sh
   npm run dev
   ```

## Project Structure

```
Plant-disease-detection/
├── .venv/                     # Virtual environment
├── fasterRCNNmodal/            # Faster R-CNN Model
├── f\Frontend/                  # Frontend source code
│   ├── src/
│   └── package.json
├── test_images/               # Sample images for testing
├── yolov8/                    # YOLOV8 Model
├── app.py                     # Main backend application
├── requirements.txt           # Python dependencies
└── README.md                  # This file

```
