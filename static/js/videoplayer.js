document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('viewer');
    const toggle = document.getElementById('togglePlay');
    const progress = document.getElementById('progress');
    const progressBar = document.querySelector('.progress__filled');
    const volumeSlider = document.getElementById('volumeSlider');
    const rateSlider = document.getElementById('playbackRateSlider');
    const currentTimeDisplay = document.getElementById('currentTime');
    const durationTimeDisplay = document.getElementById('durationTime');

    // HTML에서 전달된 highlightData를 가져옵니다.
    const highlightDataElement = document.getElementById('highlightData');
    const highlightData = JSON.parse(highlightDataElement.textContent);
    const segmentDuration = 3; // 각 길이 (초)

    // king
    const slider = document.getElementById('slider_king');

    console.log('highlightData:', highlightData);

    function togglePlay() {
        if (video.paused) {
            video.play();
        } else {
            video.pause();
        }
        updateButton();
    }

    function updateButton() {
        const icon = video.paused ? '►' : '❚❚';
        toggle.textContent = icon;
    }

    function handleRangeUpdate() {
        video[this.name] = this.value;
    }

    function handleProgress() {
        const percent = (video.currentTime / video.duration) * 100;
        progressBar.style.width = `${percent}%`;
        updateTimes();
    }

    function scrub(e) {
        const scrubTime = (e.offsetX / progress.offsetWidth) * video.duration;
        if (isBuffered(scrubTime)) {
            video.currentTime = scrubTime;
        }
    }

    function isBuffered(time) {
        const buffered = video.buffered;
        for (let i = 0; i < buffered.length; i++) {
            if (time >= buffered.start(i) && time <= buffered.end(i)) {
                return true;
            }
        }
        return false;
    }

    function formatTime(time) {
        const minutes = Math.floor(time / 60);
        const seconds = Math.floor(time % 60);
        return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
    }

    function updateTimes() {
        currentTimeDisplay.textContent = formatTime(video.currentTime);
        durationTimeDisplay.textContent = formatTime(video.duration);
    }

    function setHighlightSegments(highlightData, segmentDuration) {
        const videoDuration = video.duration;
        const segments = highlightData.length;
        const segmentWidth = 100 / segments; // 각 세그먼트의 너비 (%)

        let gradientString = 'linear-gradient(to right, ';
        highlightData.forEach((highlight, index) => {
            const startPercent = index * segmentWidth;
            const endPercent = startPercent + segmentWidth;

            if (highlight === 1) {
                gradientString += `rgba(255, 0, 0, 0.5) ${startPercent}%, rgba(255, 0, 0, 0.5) ${endPercent}%, `;
            } else if (highlight === 2) {
                gradientString += `rgba(0, 0, 255, 0.5) ${startPercent}%, rgba(0, 0, 255, 0.5) ${endPercent}%, `;
            } else {
                gradientString += `#666 ${startPercent}%, #666 ${endPercent}%, `;
            }
        });

        gradientString = gradientString.slice(0, -2) + ')'; // 마지막 쉼표 제거 후 닫는 괄호 추가
        console.log('gradientString:', gradientString); // 디버그: 최종 그라데이션 문자열 확인
        progress.style.background = gradientString;
    }

    // 이벤트 리스너 연결
    video.addEventListener('click', togglePlay);
    video.addEventListener('play', updateButton);
    video.addEventListener('pause', updateButton);
    video.addEventListener('timeupdate', handleProgress);
    video.addEventListener('loadedmetadata', () => {
        updateTimes();
        setHighlightSegments(highlightData, segmentDuration); // 하이라이트 세그먼트 설정
    });

    toggle.addEventListener('click', togglePlay);

    volumeSlider.addEventListener('change', handleRangeUpdate);
    volumeSlider.addEventListener('mousemove', handleRangeUpdate);

    rateSlider.addEventListener('change', handleRangeUpdate);
    rateSlider.addEventListener('mousemove', handleRangeUpdate);

    let mousedown = false;
    progress.addEventListener('mousedown', (e) => {
        mousedown = true;
        scrub(e);
    });
    progress.addEventListener('mouseup', () => mousedown = false);
    progress.addEventListener('mouseleave', () => mousedown = false);
    progress.addEventListener('mousemove', (e) => {
        if (mousedown) {
            scrub(e);
        }
    });

    progress.addEventListener('click', scrub);

    // Update the slider's max value when the video's metadata is loaded
    video.addEventListener('loadedmetadata', () => {
        slider.max = video.duration;
        console.log("King");
    });

    // Update the video's current time when the slider value changes
    slider.addEventListener('input', () => {
        video.currentTime = slider.value;
        console.log("HMM");
    });

    // Update the slider's value as the video plays
    video.addEventListener('timeupdate', () => {
        slider.value = video.currentTime;
        console.log("HAH");
    });
});
