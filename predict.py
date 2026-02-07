import numpy as np
import librosa
import tensorflow as tf

def extract_mel(file_path):
    y, sr = librosa.load(file_path, sr=22050, duration=3.0)
    y_trim, _ = librosa.effects.trim(y, top_db=25)
    S = librosa.feature.melspectrogram(y=y_trim, sr=sr, n_mels=128, fmax=8000)
    log_S = librosa.power_to_db(S, ref=np.max)
    target_time = 87
    if log_S.shape[1] < target_time:
        log_S = np.pad(log_S, ((0,0),(0,target_time-log_S.shape[1])))
    else:
        log_S = log_S[:, :target_time]
    return log_S

model = tf.keras.models.load_model("best_ser.keras")
classes = np.array(['angry','calm','disgust','fearful','happy','neutral','sad','surprised'])

def predict_emotion(wav_path):
    mel = extract_mel(wav_path)
    mel = mel[np.newaxis, ..., np.newaxis]
    preds = model.predict(mel)[0]
    idx = preds.argmax()
    print(f"Predicted emotion: {classes[idx]} ({preds[idx]*100:.1f}%)")

if __name__ == "__main__":
    import sys
    predict_emotion(sys.argv[1])
