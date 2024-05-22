import os
import cv2
import numpy as np

import librosa 
import librosa.display as dsp
from IPython.display import Audio
from moviepy.editor import VideoFileClip
from tensorflow.keras.models import load_model



def extract_audio(video_path, audio_path):
    '''
    1. 무비 파일 경로\n
    video_path = 'data/원천데이터/2~5분/test.mp4'
    audio_path = 'audio.wav'

    2. 오디오 추출\n
    extract_audio(video_path, audio_path)
    '''

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


def preprocess_video_every_3_seconds(video_path:str, frame_size:tuple, frame_rate=3):
    """
    Extracts frames every 3 seconds from a video file, resizing them to frame_size and converting to grayscale.
    
    Args:
    video_path (str): Path to the video file.
    frame_size (tuple): Size (height, width) to resize frames.
    frame_rate (int): Number of frames to extract per second within the 3-second window.

    Returns:
    List[numpy.ndarray]: List of sequences, where each sequence is a numpy array of shape (num_frames, height, width, 1).
    """

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
        
        if len(frames) >= frame_rate : 
            sequences.append(np.array(frames[:frame_rate * 3]))

    vidcap.release()
    return np.array(sequences[:-1])


def pipeline_video(video_path:str):

    if not os.path.exists(video_path):
        print(f"Video Not Found : {video_path}")
        return
    
    audio = extract_audio(video_path, './test.wav')
    audio = preprocess_audio(audio)

    video = preprocess_video_every_3_seconds(video_path, (256, 256), 3)

    print(len(video))
    print(len(audio))

    #video_model = load_model("video_3D_model.h5")
    video_model = load_model("video_model.h5")
    audio_model = load_model("audio_model_resnet.h5")

    video_output = video_model.predict(video)
    audio_output = audio_model.predict(audio)

    ensemble_output = np.mean([video_output, audio_output], axis=0)
    final_predictions = np.argmax(ensemble_output, axis=1)
    
    return final_predictions

video_path = "/Users/yun/Desktop/moon.mp4"
test_data = pipeline_video(video_path)

print(test_data)