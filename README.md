# Transport Emission Prediction using Logistic Regression

## Overview
This project builds a **Logistic Regression** model to predict vehicle emission test results.  
It helps identify whether a vehicle is likely to **pass** or **fail** the test based on multiple features like:
- Vehicle Age  
- Engine Size  
- Mileage per Year  
- Previous Failures  

The model is trained, tested, and evaluated using **accuracy score**, **confusion matrix**, and **classification report**.

---

## Dataset
File: `transport_emission_check.csv`

Example columns:
| Vehicle_Age | Engine_Size | Mileage_per_Year | Previous_Failures | Emission_Test_Result |
|--------------|--------------|------------------|-------------------|----------------------|
| 5 | 1.6 | 12000 | 0 | Pass |
| 8 | 2.0 | 18000 | 1 | Fail |

---

## Steps Performed
1. Imported libraries (pandas, numpy, sklearn, matplotlib).  
2. Loaded and explored dataset.  
3. Split data into **train** and **test** sets.  
4. Trained a **Logistic Regression** model.  
5. Predicted emission test results.  
6. Evaluated model performance using:
   - Accuracy Score  
   - Confusion Matrix  
   - Classification Report  
7. Visualized results using a **Confusion Matrix Heatmap**.

---

## Model Evaluation
- **Accuracy Score**: Measures overall correctness of predictions.  
- **Confusion Matrix**: Shows true vs false predictions.  
- **Classification Report**: Displays precision, recall, and F1-score.  

---

## Visualization
A confusion matrix heatmap is generated using:
```python
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
