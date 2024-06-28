# ECG Signal Analysis and Heartbeat Classification 🫀🔬

![image](https://github.com/syarwinaaa09/ecg-signal-analysis-and-heartbeat-classification/assets/114587158/a740fdf8-4d5f-4596-952e-04bba5c93400)

## What's Inside? 📦
- **Data Collection**: Grabbing some heartbeats from the MIT-BIH Arrhythmia Database.
- **Data Preprocessing**: Teaching your heartbeats some manners by filtering noise and removing artifacts.
- **QRS Detection**: Finding those QRS complexes like a heartbeat detective. 🔍
- **Classification**: Sorting your heartbeats into 'Normal' and 'Needs Attention'. 🩺
- **Model Training**: Training a Random Forest, SVM, and even a CNN because why not? 🌳🤖

## How to Use? 🚀
1. **Get the Data**:
```
wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/
```
2. **Run the Analysis**:
```
python ecg_signal_analysis.py
```
## My Results 🔬
![image](https://github.com/syarwinaaa09/ecg-signal-analysis-and-heartbeat-classification/assets/114587158/96e5e4f4-4680-48bd-9baa-32912c677303)

![image](https://github.com/syarwinaaa09/ecg-signal-analysis-and-heartbeat-classification/assets/114587158/0b946ac4-70f3-44e7-ae20-65f50dd833e8)

![image](https://github.com/syarwinaaa09/ecg-signal-analysis-and-heartbeat-classification/assets/114587158/039a4fe5-3557-436f-9bd1-4e0584c8d738)

![image](https://github.com/syarwinaaa09/ecg-signal-analysis-and-heartbeat-classification/assets/114587158/3035d494-72e7-471f-a4d4-4cac55021e7c)

![image](https://github.com/syarwinaaa09/ecg-signal-analysis-and-heartbeat-classification/assets/114587158/6df8081f-725e-4e94-aac1-60c23c6d0fb8)

![image](https://github.com/syarwinaaa09/ecg-signal-analysis-and-heartbeat-classification/assets/114587158/d424cfbd-a1cc-40af-b939-5ec5beb2eda6)

![image](https://github.com/syarwinaaa09/ecg-signal-analysis-and-heartbeat-classification/assets/114587158/effca6dc-1a84-4f9f-be23-d231858d361d)

## Acknowledgements 🙏
Big thanks to Pan and Tompkins for making QRS detection cool since 1985. 🎉

J. Pan and W. J. Tompkins, "A Real-Time QRS Detection Algorithm," in IEEE Transactions on Biomedical Engineering, vol. BME-32, no. 3, pp. 230-236, March 1985, doi: 10.1109/TBME.1985.325532.
keywords: {Detection algorithms;Electrocardiography;Detectors;Databases;Band pass filters;Interference;Filtering;Computer displays;Digital filters;Noise reduction},

## Disclaimer 🛑
This project is for educational purposes only. Do not use it to make real-life medical decisions. 

Consult a healthcare professional for serious matters. Stay healthy and keep your heart happy! ❤️
