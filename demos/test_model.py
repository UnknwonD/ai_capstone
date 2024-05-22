import os
import numpy as np
import librosa
import cv2
from moviepy.editor import VideoFileClip
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
class CustomConv2D(Conv2D):
    pass

def get_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No file or directory found at {model_path}")
    model = keras_load_model(model_path, custom_objects={'CustomConv2D': CustomConv2D})
    return model

def extract_audio(video_path, audio_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path, verbose=False, logger=None)
    video_clip.close()
    return audio_path

def preprocess_audio(audio_path, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=130, segment_duration=3):
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    segment_length = int(sr * segment_duration)
    segments = []
    num_segments = len(audio) // segment_length
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        segment = audio[start_idx:end_idx]
        mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        segments.append(mel_spectrogram_db)
    return np.array(segments)

def preprocess_video_every_3_seconds(video_path, frame_size, frame_rate=3):
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * 3)
    sequences = []
    while True:
        frames = []
        for _ in range(interval):
            success, frame = vidcap.read()
            if not success:
                break
            frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = np.expand_dims(gray_frame, axis=-1)
            gray_frame = gray_frame.astype(np.float32) / 255.0
            frames.append(gray_frame)
        if len(frames) == 0:
            break
        if len(frames) >= frame_rate:
            sequences.append(np.array(frames[:frame_rate * 3]))
    vidcap.release()
    return np.array(sequences[:-1])

def pipeline_video(video_path):
    if not os.path.exists(video_path):
        print(f"Video Not Found : {video_path}")
        return
    audio = extract_audio(video_path, './test.wav')
    audio = preprocess_audio(audio)
    video = preprocess_video_every_3_seconds(video_path, (256, 256), 3)
    video_model = get_model("video_model.h5")
    audio_model = get_model("audio_model_resnet.h5")
    video_output = video_model.predict(video)
    audio_output = audio_model.predict(audio)
    ensemble_output = np.mean([video_output, audio_output], axis=0)
    final_predictions = np.argmax(ensemble_output, axis=1)
    return final_predictions

if __name__ == "__main__":
    video_path = "path_to_your_video.mp4"
    predictions = pipeline_video(video_path)
    print("Predictions:", predictions)
