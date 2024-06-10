from django.shortcuts import render, redirect, get_object_or_404
import os
import platform

if platform.system() == "Darwin":
    os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
    os.environ['FFMPEG_BINARY'] = "/opt/homebrew/bin/ffmpeg"

from .models import Video
from .forms import VideoForm
from django.conf import settings  
import cv2
import json
import numpy as np
import librosa
from moviepy.editor import VideoFileClip
import tensorflow as tf
from keras.models import load_model
import threading
import queue
from datetime import datetime
from django.http import JsonResponse
from django.core.cache import cache
import glob
from pathlib import Path


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

def get_max_values_and_indices(video_data, audio_data, video_weight, audio_weight, threshold, video_length, ratio):
    
    min_length = min(video_data.shape[0], audio_data.shape[0])
    video_data = video_data[:min_length]
    audio_data = audio_data[:min_length]
    
    if video_length < 0:
        output_length = int(min_length * ratio)
    else:
        output_length = (video_length // 3)
    
    ensemble_scores = (video_data * video_weight + audio_data * audio_weight) / (video_weight + audio_weight)
    ensemble_labels = ensemble_scores.argmax(axis=1)

    high_confidence_twos = ensemble_scores[:, 2] >= threshold
    ensemble_labels[high_confidence_twos] = 2
    
    output = [(i, ensemble_labels[i], max(ensemble_scores[i])) for i in range(min_length)]
    
    sorted_data = sorted(output, key=lambda x: (x[1], x[2]), reverse=True)
    sorted_data = sorted(sorted_data[:output_length], key=lambda x: x[0])
    
    return sorted_data

def preprocess_shorts_only_frame(video_path: str, label: list, output_path: str):
    vidcap = cv2.VideoCapture(video_path)
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    
    interval = int(fps * 3)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    sequences = []
    
    for lbl in label:
        index = lbl[0]
        start_frame = float(index * interval)

        if start_frame >= total_frames:
            continue
        
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_pos = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
        
        if current_pos != start_frame:
            continue

        frames = []
        for _ in range(interval):
            success, frame = vidcap.read()
            if not success:
                break
            
            frames.append(frame)
        
        if len(frames) == interval:
            sequences.extend(frames)
    
    if sequences:
        height, width, layers = sequences[0].shape
        size = (width, height)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

        for frame in sequences:
            out.write(frame)
        
        out.release()
    
    vidcap.release()
    
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

    video_weight = 0.65  # 초기값
    audio_weight = 1 - video_weight
    threshold = 0.7

    ensemble_output, ensemble_scores = compute_ensemble(new_video_data, new_audio_data, video_weight, audio_weight, threshold)
    print('ensemble_output:', ensemble_output)

    # Cache the processed data
    cache.set('new_video_data', new_video_data, timeout=300)  # Cache for 5 minutes
    cache.set('new_audio_data', new_audio_data, timeout=300)  # Cache for 5 minutes

    return ensemble_output

#처음 프로세스 이후에 display_video.html에서 입력했을 때 모델 작동
def process_video_data_after(video_path, video_weight, threshold, video_length, ratio):
    new_video_data = cache.get('new_video_data')
    new_audio_data = cache.get('new_audio_data')
    print("cache Contents")
    print(new_video_data)
    print("cached")

    if new_video_data is None or new_audio_data is None:
        return JsonResponse({'error': 'Cached data not found. Please reprocess the video.'}, status=400)

    audio_weight = 1 - video_weight
    

    sorted_data = get_max_values_and_indices(new_video_data, new_audio_data, video_weight, audio_weight, threshold, video_length, ratio)

    current_time = str(datetime.now().strftime("%Y%m%d_%H%M%S")) + ".mp4"
    output_path = os.path.join("/Users/idaeho/Documents/GitHub/ai_capstone/media/output_video", current_time)

    print("video processing")
    preprocess_shorts_only_frame(video_path, sorted_data, output_path)

    print('ensemble_output:', sorted_data)

    ensemble_output, ensemble_scores = compute_ensemble(new_video_data, new_audio_data, video_weight, audio_weight, threshold)
    print('ensemble_output:', ensemble_output)

    return ensemble_output


def upload_video(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save()
            video_file_path = video.input_video.path

            # Extract and save the first frame of the video
            first_frame_path = os.path.join(settings.MEDIA_ROOT, f'frame_{video.id}.jpg')
            extract_first_frame(video_file_path, first_frame_path)

            predictions = process_video_data(video_file_path)
            highlight_data_json = json.dumps(predictions, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))

            # Fetch latest videos and their first frames
            video_dir = os.path.join(settings.MEDIA_ROOT, 'output_video')
            latest_videos = get_latest_videos(video_dir, 2)

            first_frame_paths = []
            for video_path in latest_videos:
                video_name = Path(video_path).stem
                first_frame_path = os.path.join(settings.MEDIA_ROOT, f'{video_name}.jpg')
                extract_first_frame(video_path, first_frame_path)
                first_frame_paths.append(settings.MEDIA_URL + f'{video_name}.jpg')

            return render(request, 'display_video.html', {
                'video': video,
                'highlightData': highlight_data_json,
                'first_frame_url': settings.MEDIA_URL + f'frame_{video.id}.jpg',  # URL for the first frame image
                'first_frame_urls': first_frame_paths  # First frames of the latest videos
            })
    else:
        form = VideoForm()

        # Fetch latest videos and their first frames for the homepage
        video_dir = os.path.join(settings.MEDIA_ROOT, 'output_video')
        latest_videos = get_latest_videos(video_dir, 5)

        first_frame_paths = []
        for video_path in latest_videos:
            video_name = Path(video_path).stem
            first_frame_path = os.path.join(settings.MEDIA_ROOT, f'{video_name}.jpg')
            extract_first_frame(video_path, first_frame_path)
            first_frame_paths.append(settings.MEDIA_URL + f'{video_name}.jpg')

        return render(request, 'upload_video.html', {
            'form': form,
            'first_frame_urls': first_frame_paths  # First frames of the latest videos
        })

def display_video_thumbnails(request):
    video_dir = '/Users/idaeho/Documents/GitHub/ai_capstone/media/output_video'
    latest_videos = get_latest_videos(video_dir, 2)

    # Extract and save the first frame of each video
    first_frame_paths = []
    for video_path in latest_videos:
        video_name = Path(video_path).stem
        first_frame_path = os.path.join(settings.MEDIA_ROOT, f'{video_name}.jpg')
        extract_first_frame(video_path, first_frame_path)
        first_frame_paths.append(settings.MEDIA_URL + f'{video_name}.jpg')

    return render(request, 'display_video.html', {
        'first_frame_urls': first_frame_paths
    })


def display_video(request, video_id):
    video = get_object_or_404(Video, pk=video_id)
    return render(request, 'display_video.html', {'video': video})

#가중치 프로세싱
def process_weights(request, video_id):
    if request.method == 'POST':
        video = get_object_or_404(Video, pk=video_id)
        video_file_path = video.input_video.path
        # video_weight = float(request.POST.get('video_weight', 0.5))
        # threshold = float(request.POST.get('threshold', 0.7))
        
        # Retrieve selected video categories

        video_categories = request.POST.get('video_category')
        video_length = request.POST.get('video_length')
        
        ratio = 0
        
        print(video_categories)
        
        if video_categories == 'shorts':
            video_length = int(video_length)
            video_weight = 0.45
            threshold = 0.6
            
        elif video_categories == 'summary':
            video_length = int(video_length)
            video_weight = 0.35
            threshold = 0.5
            
        else:
            ratio = float(video_length)
            video_length = -1
            video_weight = 0.55
            threshold = 0.6


        # Process predictions based on categories
        predictions = process_video_data_after(video_file_path, video_weight, threshold, video_length, ratio)
        print("ggg")
        
        highlight_data_json = json.dumps(predictions, default=convert_to_serializable)
        print("ddd")

        return render(request, 'display_video.html', {
            'video': video,
            'highlightData': highlight_data_json,
            'video_weight': video_weight,
            'threshold': threshold,
            'video_categories': video_categories,
        })
    return JsonResponse({'error': 'Invalid request'}, status=400)


def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
    
    
    
#썸네일 뽑기 위한 최근 동영상 2개
def get_latest_videos(video_dir, n=2):
    list_of_files = glob.glob(os.path.join(video_dir, '*.mp4'))
    latest_files = sorted(list_of_files, key=os.path.getmtime, reverse=True)[:n]
    return latest_files

def extract_first_frame(video_path, output_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    if success:
        cv2.imwrite(output_path, image)
    vidcap.release()
