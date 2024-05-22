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

def pipeline_video(video_path):
    audio_path = './test.wav'
    audio = extract_audio(video_path, audio_path)
    audio_processed = preprocess_audio(audio)
    
    # 모델을 이용한 예측 로직
    # video_predictions = video_model.predict(...)  # 비디오 모델 예측 로직 추가
    audio_predictions = audio_model.predict(audio_processed)  # 오디오 모델 예측 로직 추가
    
    # 결과 처리
    return audio_predictions  # 예시로 오디오 처리 결과만 반환

# 뷰 정의
def upload_video(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save()
            video_file_path = video.input_video.path

            # AI 영상 처리 함수 호출
            predictions = pipeline_video(video_file_path)
            highlightData = process_predictions_to_highlights(predictions)  # 데이터 처리 함수

            # JSON 형식으로 변환
            highlight_data_json = json.dumps(highlightData)
            print('predictions', predictions)

            return render(request, 'display_video.html', {
                'video': video,
                'highlightData': highlight_data_json
            })
    else:
        form = VideoForm()
    return render(request, 'upload_video.html', {'form': form})

def process_predictions_to_highlights(predictions):
    # 예측 결과를 분석하여 하이라이트 데이터 생성
    # 각 예측의 argmax를 이용하여 하이라이트 여부를 결정
    highlights = [int(np.argmax(score)) for score in predictions]
    print('highlights:', highlights)
    return highlights

def display_video(request, video_id):
    video = get_object_or_404(Video, pk=video_id)
    return render(request, 'display_video.html', {'video': video})
