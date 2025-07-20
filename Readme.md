# **Text Extraction from Product Labels 🏷️**

This project provides a complete pipeline for extracting specific text fields—**GTIN, Serial Number, LOT Number, and Expiry Date**—from product label images. It leverages a deep learning approach, combining object detection with optical character recognition (OCR) for accurate and automated data extraction.

## **🚀 Overview**

The core of this project is a two-stage process:

1. **Object Detection:** A **YOLOv8-obb** (Oriented Bounding Box) model is trained to identify and locate the precise regions of the four target text fields on an image.  
2. **Text Recognition:** **PaddleOCR** is then used to perform OCR on these specific regions to extract the text content.

The models are also converted to **TensorRT** for optimized inference performance on NVIDIA GPUs.

## **✨ Features**

* **Targeted Extraction:** Specifically trained to find and read:  
  * GTIN (Global Trade Item Number)  
  * SR\_NO (Serial Number)  
  * LOT (Lot Number)  
  * EXP (Expiry Date)  
* **High Accuracy:** The YOLOv8 model is trained on a custom dataset to achieve robust detection.  
* **Optimized for Speed:** Includes scripts for converting models to TensorRT, significantly speeding up inference time.  
* **End-to-End Pipeline:** From model training to final text extraction and saving results to a CSV file.

## **📂 Project Structure**

Text-Extraction-Project/  
├── extracted\_results/  
│   ├── extracted\_data.csv        \# Final extracted text output  
│   └── processing\_summary.txt    \# Summary of the image processing run  
├── runs/  
│   └── obb/train/  
│       ├── args.yaml             \# YOLO training configuration  
│       ├── results.csv           \# Training metrics per epoch  
│       └── weights/  
│           └── best.pt           \# Best trained YOLO model weights  
├── data.yaml                     \# Dataset configuration for YOLO  
├── PaddleOCR Model.ipynb         \# Notebook for OCR logic  
├── Requirements.txt              \# Project dependencies  
├── TensorRT conversion YOLO.ipynb \# Script for YOLO to TensorRT conversion  
├── TensorRT deployment.ipynb     \# Script for running the optimized models  
├── Untitled.ipynb                \# Main notebook for batch image processing  
└── YOLO Training.ipynb           \# Notebook for training the YOLOv8 model

## **⚙️ Setup and Installation**

**⚠️ Important:** This project requires a specific environment with a GPU and correctly configured NVIDIA libraries. The current version has known dependency issues.

### **1\. Prerequisites**

* NVIDIA GPU  
* NVIDIA Driver  
* CUDA Toolkit  
* cuDNN  
* TensorRT

It is **critical** that the versions of CUDA, cuDNN, and TensorRT are compatible with your version of PaddlePaddle-GPU. Please refer to the official PaddlePaddle documentation for version compatibility. The errors in this project (cudnn64\_8.dll not found) suggest a mismatch between these libraries.

### **2\. Clone the Repository**

git clone \<repository-url\>  
cd Text-Extraction-Project

### **3\. Install Python Dependencies**

It is highly recommended to use a virtual environment (e.g., venv or conda).

python \-m venv venv  
source venv/bin/activate  \# On Windows use \`venv\\Scripts\\activate\`  
pip install \-r Requirements.txt

## **🛠️ Usage and Workflow**

### **1\. Train the Object Detection Model**

* **Data Preparation:** Organize your labeled dataset and update the data.yaml file with the correct paths for training, validation, and test sets.  
* **Run Training:** Open and execute the YOLO Training.ipynb notebook. The training process will run for 100 epochs by default. The best model weights (best.pt) will be saved in the runs/obb/train/weights/ directory.

### **2\. Convert Models to TensorRT (Optional, for performance)**

* **YOLO Conversion:** Run the TensorRT conversion YOLO.ipynb notebook to convert the best.pt model into a .engine file for faster inference.  
* **PaddleOCR Conversion:** Run the TensorRT conversion PaddleOCR.ipynb notebook to do the same for the PaddleOCR models.

### **3\. Run the Extraction Pipeline**

* **Configure Paths:** Open the Untitled.ipynb notebook. This is the main script for processing images.  
* **Set Model Path:** In the ProductLabelReader class, ensure the model\_path points to your trained YOLO model (best.pt or the converted .engine file).  
* **Set Image Directory:** Specify the path to the directory containing the images you want to process.  
* **Execute:** Run all cells in the notebook. The script will:  
  1. Detect text regions in each image.  
  2. Crop these regions.  
  3. Use PaddleOCR to extract the text.  
  4. Clean the extracted text.  
  5. Save the final results in extracted\_results/extracted\_data.csv.

## **📊 Results**

The primary output is extracted\_results/extracted\_data.csv, a table containing the filename and the extracted text for each of the four fields.

The YOLOv8-obb model was trained for 100 epochs and achieved a **mean Average Precision (mAP50-95) of 0.6918**, indicating a good performance in detecting the text regions.

## **❗ Known Issues & Future Work**

### **Current Issues**

* **Dependency Errors:** The project currently fails during the OCR step due to issues with NVIDIA library paths. Errors like RuntimeError: TensorRT dynamic library is not found and PreconditionNotMet: Could not find registered platform with id: 0x... point to problems with the CUDA/cuDNN/TensorRT installation or environment variables.  
  * **Solution:** Carefully reinstall NVIDIA libraries, ensuring they are compatible with your PyTorch and PaddlePaddle versions. Make sure the library paths are correctly set in your system's environment variables.

### **Future Improvements**

* **Robust Error Handling:** Implement more specific error handling to gracefully manage images where text cannot be found or read.  
* **Text Validation:** Use regular expressions to validate the format of the extracted text (e.g., ensure EXP is a valid date format, GTIN contains only numbers).  
* **Environment Dockerization:** Create a Dockerfile to encapsulate the entire environment, making it much easier to replicate and run the project without dependency headaches.  
* **Streamlined Scripting:** Combine the Jupyter notebooks into a single, modular Python script that can be run from the command line with arguments for the image directory and model paths.