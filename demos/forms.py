from django import forms
from .models import Video

#forms.py 파일을 만들어 파일 업로드를 위한 폼을 정의
class VideoForm(forms.ModelForm):
    class Meta:
        model = Video
        fields = ['input_video']
