from django.shortcuts import render, redirect, get_object_or_404
from .models import Video
from .forms import VideoForm
import os
import cv2
import json
import numpy as np
import librosa
from moviepy.editor import VideoFileClip
import tensorflow as tf
from keras.models import load_model
import threading
import queue
from django.http import JsonResponse

# 전역 변수로 모델을 로드합니다.
video_model = load_model("video_3D_model.h5")
audio_model = load_model("audio_comp_model.h5")

# AI 모델 처리 함수
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

def preprocess_video_every_3_seconds(video_path: str, frame_size: tuple, frame_rate=3):
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * 3)
    target_frames = int(frame_rate * 3)
    sequences = []

    def read_frames(q):
        while True:
            success, frame = vidcap.read()
            if not success:
                q.put(None)
                break
            q.put(frame)

    frame_queue = queue.Queue(maxsize=100)
    threading.Thread(target=read_frames, args=(frame_queue,)).start()

    while True:
        frames = []
        for _ in range(interval_frames):
            frame = frame_queue.get()
            if frame is None:
                break
            frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = np.expand_dims(gray_frame, axis=-1)
            gray_frame = gray_frame.astype(np.float32) / 255.0
            frames.append(gray_frame)
        
        if len(frames) < interval_frames:
            break
        
        sequences.append(np.array(frames[:target_frames]))
    
    vidcap.release()
    return np.array(sequences)

def pipeline_video(video_path:str):
    audio_path = './test.wav'
    audio = extract_audio(video_path, audio_path)
    audio = preprocess_audio(audio)

    video = preprocess_video_every_3_seconds(video_path, (256, 256), 3)

    video_model = load_model("video_3D_model.h5")
    audio_model = load_model("audio_comp_model.h5")

    video_output = video_model.predict(video)
    audio_output = audio_model.predict(audio)
    
    return video_output, audio_output

def compute_ensemble(video_data, audio_data, video_weight, audio_weight, threshold):
    min_length = min(video_data.shape[0], audio_data.shape[0])
    video_data = video_data[:min_length]
    audio_data = audio_data[:min_length]
    
    ensemble_scores = (video_data * video_weight + audio_data * audio_weight) / (video_weight + audio_weight)
    ensemble_labels = ensemble_scores.argmax(axis=1)

    high_confidence_twos = ensemble_scores[:, 2] >= threshold
    ensemble_labels[high_confidence_twos] = 2
    
    return ensemble_labels, ensemble_scores

def process_video_data(video_path):
    video_data, audio_data = pipeline_video(video_path)

    new_audio_data = np.zeros((audio_data.shape[0], audio_data.shape[1] + 1))
    for i, audio_row in enumerate(audio_data):
        half_value = audio_row[1] / 2
        new_audio_data[i][0] = round(audio_row[0], 5)
        new_audio_data[i][1] = round(half_value, 5)
        new_audio_data[i][2] = round(half_value, 5)

    new_video_data = np.round(video_data, 5)

    video_weight = 0.5  # 초기값
    audio_weight = 1 - video_weight
    threshold = 0.8
    

    ensemble_output, ensemble_scores = compute_ensemble(new_video_data, new_audio_data, video_weight, audio_weight, threshold)
    print('ensemble_output:', ensemble_output)
    return ensemble_output


#처음 프로세스 이후에 display_video.html에서 입력했을 때 모델 작동
def process_video_data_after(video_path, video_weight, threshold):
    video_data, audio_data = pipeline_video(video_path)

    new_audio_data = np.zeros((audio_data.shape[0], audio_data.shape[1] + 1))
    for i, audio_row in enumerate(audio_data):
        half_value = audio_row[1] / 2
        new_audio_data[i][0] = round(audio_row[0], 5)
        new_audio_data[i][1] = round(half_value, 5)
        new_audio_data[i][2] = round(half_value, 5)

    new_video_data = np.round(video_data, 5)
    audio_weight = 1 - video_weight

    ensemble_output, ensemble_scores = compute_ensemble(new_video_data, new_audio_data, video_weight, audio_weight, threshold)
    print('ensemble_output after:', ensemble_output)
    return ensemble_output

def upload_video(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save()
            video_file_path = video.input_video.path

            predictions = process_video_data(video_file_path)
            highlight_data_json = json.dumps(predictions.tolist())

            return render(request, 'display_video.html', {
                'video': video,
                'highlightData': highlight_data_json
            })
    else:
        form = VideoForm()
    return render(request, 'upload_video.html', {'form': form})

def display_video(request, video_id):
    video = get_object_or_404(Video, pk=video_id)
    return render(request, 'display_video.html', {'video': video})

#가중치 프로세싱
def process_weights(request, video_id):
    if request.method == 'POST':
        video = get_object_or_404(Video, pk=video_id)
        video_file_path = video.input_video.path
        video_weight = float(request.POST.get('video_weight', 0.5))
        threshold = float(request.POST.get('threshold', 0.7))

        predictions = process_video_data_after(video_file_path, video_weight, threshold)
        highlight_data_json = json.dumps(predictions.tolist())

        return render(request, 'display_video.html', {
            'video': video,
            'highlightData': highlight_data_json,
            'video_weight': video_weight,
            'threshold': threshold,
        })
    return JsonResponse({'error': 'Invalid request'}, status=400)
