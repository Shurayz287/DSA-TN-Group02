# 🧠 DSA-TN-Group02 — Duplicate Image Detection Project

## 📘 Introduction
This repository is the official project of **Group 02 – Talented Program in Computer Science, Cohort K24**, Ho Chi Minh City University of Technology (HCMUT).  
The project aims to **detect and remove duplicate or near-duplicate images** from large datasets by combining various **image hashing algorithms** and **FAISS** similarity search.  

### 👨‍💻 Team Members
- **Vu Hoang Hai**  
- **Nguyen Ngoc Thach**
- **Nguyen Hoang Gia Huy**  
---

## 🚀 Project Pipeline Overview

The duplicate detection system is built as a modular pipeline consisting of several main stages:

### 1️⃣ **Preprocessing**
- Load all images from the input folder.  
- Normalize image size and format.  
- Remove **exact duplicates** using **MD5 Hashing** (bitwise identical check).

### 2️⃣ **Feature Extraction**
Each image is converted into a compact numerical signature using different algorithms:
- **AverageHash**, **PerceptualHash**, **DifferenceHash** (via `imagehash` library)
- **SimHash** – converts brightness values into a 64-bit signature.  
- **MinHash** – tokenizes brightness patterns and applies Locality-Sensitive Hashing (LSH).  
- **FAISS (Facebook AI Similarity Search)** – compares feature vectors using cosine or L2 distance metrics.

### 3️⃣ **Clustering / Grouping**
- Images with high similarity are grouped into clusters.  
- Each cluster’s **representative image** is chosen as the one with the **highest resolution**.

### 4️⃣ **Performance Evaluation**
- Evaluate algorithm accuracy with **Precision**, **Recall**, and **F1-score**.  
- Measure **execution time**, **memory usage**, and the **number of duplicates removed**.

### 5️⃣ **Reporting**
- Print detailed summary statistics:  
  total images, clusters, representatives, and performance metrics.  
- Export representative (clean) images into the `Cleaned/` directory.

## ⚙️ How to Run: Demo on Google Colab