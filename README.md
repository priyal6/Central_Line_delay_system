# Central_Line_delay_system
# 🚇 Central Line Delay Predictor

[![Live Site](https://img.shields.io/badge/Live%20Site-Click%20Here-brightgreen?style=for-the-badge)](https://centralline.netlify.app/)

A real-time machine learning project that predicts **train delays** on the **London Central Line** using live data from the **Transport for London (TfL) Unified API**

---

## 📌 Overview


- 🔄 Collects real-time Central Line train data
- 🧹 Cleans and processes the data
- 🧠 Trains a classification model to predict delays (>2 minutes)
- 📈 Visualizes performance and feature importance
- 🌐 Hosts a live dashboard showing the latest predictions

---

## 🛠 Tech Stack

- **Python** – Core programming logic
- **Pandas, Scikit-learn, LightGBM** – Machine learning
- **Seaborn, Matplotlib** – Data visualization
- **Requests** – TfL API integration
- **Netlify** – Frontend deployment
- **TfL Unified API** – Live train prediction data

---

## 🔍 Problem Statement

aim to predict whether an incoming Central Line train will be **delayed by more than 2 minutes** based on:

- **Station name**
- **Time of request**
- **Platform name**
- **Scheduled vs actual arrival gap**

---

## 🧠 Prediction Logic

A train is considered **delayed** if:

```python
expectedArrival - timestamp > 2 minutes
