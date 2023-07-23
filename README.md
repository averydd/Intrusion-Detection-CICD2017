# Intrusion Detection CICD2017
## Overview
Intrusion Detection is a critical component of network security, aimed at identifying and responding to unauthorized or malicious activities within a computer network. The objective of this project is to develop an effective Intrusion Detection System (IDS) using three popular machine learning classifiers: K-Nearest Neighbors (KNN), Random Forest, and AdaBoost. By leveraging these classifiers, the project aims to build a robust and accurate IDS capable of distinguishing normal network behavior from potential intrusions.
## Set up
1. Get Data:
- Intrusion Detection using Machine Learning on CIC Dataset 2017 CSV files are to be found here:
https://drive.google.com/file/d/1-uwoKddOHgRxS8vth-nGBqBtz-qzRSAX/view
- Raw data can be dowloaded from: https://www.unb.ca/cic/datasets/ids-2017.html
2. Data preparation


| Step              | Package Requirements                                   | Compilation Command                                                                                                    |
|-------------------|--------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| Pre-Process Data  | OpenCV, NumPy, Pandas, Sklearn                         | `python Preprocessing.py`                                                                                              |
| Attack Filtering  | OpenCV, Pandas, Random, Time                           | `python Attack_Filtering.py`                                                                                           |
| Feature Selection | OpenCV, NumPy, Pandas, Sklearn, Time, Matplotlib       | `python Feature_Selection.py` (For all data) and `python Feature_Selection_Attack_Files.py`  (For each type of Attack) |
                                                                            |

Note: Make sure to have the required packages installed in your Python environment before executing the compilation commands. Additionally, ensure that you have completed the pre-processing and attack filtering steps as prerequisites for feature selection.
3. Train model:
- Package: OpenCV, NumPy, Pandas, Sklearn, Time, Matplotlib, Warning, Math
- Compile these filles: ` python ML_7_Features.py`, ` python ML_18_Features.py`, ` python ML_Attack_Features.py`, ` python ML_Final_Features.py`
## Collaborators
| Name             | Student ID | Email                               |
|------------------|------------|-------------------------------------|
| Vu Tung Linh     | 20210523   | Linh.VT210523@sis.hust.edu.vn        |
| Ta Quang Duy     | 20214884   | Duy.TQ214884@sis.hust.edu.vn         |
| Dao Ha Xuan Mai  | 20210562   | Mai.DHX20210562@sis.hust.edu.vn      |

If you have any questions or encounter any issues, please feel free to contact us through the provided email addresses.
