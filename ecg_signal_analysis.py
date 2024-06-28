import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Input
from keras.utils import to_categorical
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

# set the environment variable to turn off onednn custom operations if needed
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_ecg_record(record_path):
    record = wfdb.rdrecord(record_path)
    annotations = wfdb.rdann(record_path, 'atr')
    signal = record.p_signal[:, 0]  # get the first channel (lead)
    fs = record.fs  # sampling frequency
    return signal, annotations, fs

def bandpass_filter(signal, lowcut, highcut, fs, order=1):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def highpass_filter(signal, cutoff, fs, order=1):
    nyquist = 0.5 * fs
    high = cutoff / nyquist
    b, a = butter(order, high, btype='high')
    return filtfilt(b, a, signal)

def notch_filter(signal, notch_freq, fs, quality_factor=30):
    nyquist = 0.5 * fs
    notch = notch_freq / nyquist
    b, a = butter(2, [notch - 0.005, notch + 0.005], btype='bandstop')
    return filtfilt(b, a, signal)

def preprocess_signal(signal, fs, lowcut=0.5, highcut=50.0, baseline_cutoff=0.5, notch_freq=60.0):
    filtered_signal = bandpass_filter(signal, lowcut, highcut, fs)
    signal_without_baseline_wander = highpass_filter(filtered_signal, baseline_cutoff, fs)
    return notch_filter(signal_without_baseline_wander, notch_freq, fs)

def pan_tompkins_detector(signal, fs):
    bandpass_signal = bandpass_filter(signal, 5, 20, fs)
    derivative_signal = np.diff(bandpass_signal, prepend=bandpass_signal[0])
    squared_signal = derivative_signal ** 2
    window_size = int(0.12 * fs)  # 120 ms window
    integrated_signal = np.convolve(squared_signal, np.ones(window_size) / window_size, mode='same')
    threshold = np.mean(integrated_signal) + 0.8 * np.std(integrated_signal)
    peaks, _ = find_peaks(integrated_signal, distance=int(0.25 * fs), height=threshold)
    return peaks

def extract_features(signal, qrs_peaks, fs):
    rr_intervals = np.diff(qrs_peaks) / fs
    qrs_durations = []
    qrs_amplitudes = []

    for i in range(len(qrs_peaks) - 1):
        qrs_start = qrs_peaks[i]
        qrs_end = qrs_peaks[i + 1]
        qrs_duration = (qrs_end - qrs_start) / fs
        qrs_amplitude = np.max(signal[qrs_start:qrs_end]) - np.min(signal[qrs_start:qrs_end])
        qrs_durations.append(qrs_duration)
        qrs_amplitudes.append(qrs_amplitude)

    qrs_durations = np.array(qrs_durations)
    qrs_amplitudes = np.array(qrs_amplitudes)

    min_length = min(len(rr_intervals), len(qrs_durations), len(qrs_amplitudes))
    rr_intervals = rr_intervals[:min_length]
    qrs_durations = qrs_durations[:min_length]
    qrs_amplitudes = qrs_amplitudes[:min_length]

    return np.vstack([rr_intervals, qrs_durations, qrs_amplitudes]).T

def get_labels(annotation):
    labels = [1 if annotation.symbol[i] != 'N' else 0 for i in range(len(annotation.sample) - 1)]
    return labels

def plot_ecg_signals(original_signal, processed_signal, qrs_peaks, fs):
    time = np.arange(len(original_signal)) / fs

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(time, original_signal)
    plt.title('Original Signal')
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 2)
    plt.plot(time, processed_signal)
    plt.title('Processed Signal')
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 3)
    plt.plot(time, processed_signal, label='Processed ECG Signal')
    plt.plot(qrs_peaks / fs, processed_signal[qrs_peaks], 'ro', label='Detected QRS Complexes')
    plt.title('QRS Detection')
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# load and preprocess the ECG record
signal, annotation, fs = load_ecg_record('ecg_datasets/mit-bih/100')
processed_signal = preprocess_signal(signal, fs)

# detect QRS complexes
qrs_peaks = pan_tompkins_detector(processed_signal, fs)

# plot the original signal with detected QRS complexes
plot_ecg_signals(signal, processed_signal, qrs_peaks, fs)

# extract features and get labels
features = extract_features(processed_signal, qrs_peaks, fs)
labels = get_labels(annotation)
labels = labels[:len(features)]

# standardize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# apply smote to balance the classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(features, labels)

# split resampled data into training and testing sets
X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_resampled), y=y_train_resampled)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

# random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights_dict)
rf_classifier.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf_classifier.predict(X_test_resampled)
print("Random Forest Classification Report (Resampled Data):")
print(classification_report(y_test_resampled, y_pred_rf, zero_division=0))

# support vector machine classifier
svm_classifier = SVC(kernel="linear", random_state=42, class_weight=class_weights_dict)
svm_classifier.fit(X_train_resampled, y_train_resampled)
y_pred_svm = svm_classifier.predict(X_test_resampled)
print("SVM Classification Report (Resampled Data):")
print(classification_report(y_test_resampled, y_pred_svm, zero_division=0))

# convolutional neural network (cnn) classifier
X_train_cnn = X_train_resampled.reshape(X_train_resampled.shape[0], X_train_resampled.shape[1], 1)
X_test_cnn = X_test_resampled.reshape(X_test_resampled.shape[0], X_test_resampled.shape[1], 1)
y_train_cnn = to_categorical(y_train_resampled)
y_test_cnn = to_categorical(y_test_resampled)

cnn_classifier = Sequential([
    Input(shape=(X_train_cnn.shape[1], 1)),
    Conv1D(64, 2, activation='relu'),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(y_train_cnn.shape[1], activation='softmax')
])

cnn_classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_classifier.fit(X_train_cnn, y_train_cnn, epochs=10, batch_size=32, verbose=1)

# evaluate cnn
loss, accuracy = cnn_classifier.evaluate(X_test_cnn, y_test_cnn, verbose=0)
print("CNN Accuracy (Resampled Data): {:.2f}%".format(accuracy * 100))

# load and preprocess a new ecg record
new_signal, _, new_fs = load_ecg_record('ecg_datasets/mit-bih/101')
new_processed_signal = preprocess_signal(new_signal, new_fs)
new_qrs_peaks = pan_tompkins_detector(new_processed_signal, new_fs)
new_features = extract_features(new_processed_signal, new_qrs_peaks, new_fs)
new_features = scaler.transform(new_features)

# predictions for new data
new_rf_pred = rf_classifier.predict(new_features)
print("Random Forest Predictions for New Data:", new_rf_pred)

new_svm_pred = svm_classifier.predict(new_features)
print("SVM Predictions for New Data:", new_svm_pred)

new_features_cnn = new_features.reshape(new_features.shape[0], new_features.shape[1], 1)
new_cnn_pred = cnn_classifier.predict(new_features_cnn)
new_cnn_pred_classes = np.argmax(new_cnn_pred, axis=1)
print("CNN Predictions for New Data:", new_cnn_pred_classes)

# plot confusion matrices
plot_confusion_matrix(y_test_resampled, y_pred_rf, 'Random Forest')
plot_confusion_matrix(y_test_resampled, y_pred_svm, 'SVM')
cnn_test_pred = cnn_classifier.predict(X_test_cnn)
cnn_test_pred_classes = np.argmax(cnn_test_pred, axis=1)
plot_confusion_matrix(np.argmax(y_test_cnn, axis=1), cnn_test_pred_classes, 'CNN')