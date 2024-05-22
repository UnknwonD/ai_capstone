from django.db import models

#models.py 파일에 두 개의 영상 파일 필드를 포함하는 모델을 생성
#하나는 사용자가 업로드한 영상을 저장하고, 다른 하나는 AI가 생성한 영상을 저장


class Video(models.Model):
    input_video = models.FileField(upload_to='videos/')
    output_video = models.FileField(upload_to='processed_videos/', null=True, blank=True)

    def __str__(self):
        return f"Video {self.pk}"

