"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from demos import views  # demos 앱의 views 모듈에서 가져오기

urlpatterns = [
    path('', views.upload_video, name='home'),  # 홈페이지로 설정
    path('upload/', views.upload_video, name='upload_video'),
    path('video/<int:video_id>/', views.display_video, name='display_video'),
    path('process_weights/<int:video_id>/', views.process_weights, name='process_weights'),
    path('thumbnails/', views.display_video_thumbnails, name='display_video_thumbnails'),

]

from django.conf import settings
from django.conf.urls.static import static

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
