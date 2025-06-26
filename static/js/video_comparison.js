
// Variables for keeping track of current video and mode (RGB/Mesh)
let currentVideoIndex = 1; // Starts at video 1
const totalVideos = 10;
let mode = "mesh"; 

// Select buttons and video elements
const toggleVideoComparisonLeftButton = document.querySelector('#toggleVideoComparison .toggle-left');
const toggleVideoComparisonRightButton = document.querySelector('#toggleVideoComparison .toggle-right');
const prevVideo = document.getElementById('prevVideo');
const nextVideo = document.getElementById('nextVideo');
const videoElement = document.getElementById('video-compare');
const videoSource = document.getElementById('video-source');
const videoCounter = document.getElementById('videoCounter');

function updateVideoSource() {
  const videoIndexStart = String((currentVideoIndex - 1) * 10 + 1).padStart(4, '0');
  const videoIndexEnd = String(currentVideoIndex * 10).padStart(4, '0');

  const videoPath = `./static/videos/compare_videos/${videoIndexStart}_to_${videoIndexEnd}_${mode}_crf28.mp4`;

  videoSource.src = videoPath;
  videoElement.load();

  image_index_start = String((currentVideoIndex - 1) * 10 + 1);
  image_index_end = String(currentVideoIndex * 10)
  videoCounter.innerText = `Image ${image_index_start}-${image_index_end} of 100`;
}


toggleVideoComparisonRightButton.addEventListener('click', () => {
  mode = "rgb"; 
  toggleVideoComparisonLeftButton.classList.remove('active');
  toggleVideoComparisonRightButton.classList.add('active');
  updateVideoSource(); 
});


toggleVideoComparisonLeftButton.addEventListener('click', () => {
  mode = "mesh"; 
  toggleVideoComparisonLeftButton.classList.add('active');
  toggleVideoComparisonRightButton.classList.remove('active');
  updateVideoSource(); 
});


nextVideo.addEventListener('click', () => {
  if (currentVideoIndex < totalVideos) {
    currentVideoIndex++;
    updateVideoSource(); 
  }
});


prevVideo.addEventListener('click', () => {
  if (currentVideoIndex > 1) {
    currentVideoIndex--;
    updateVideoSource(); 
  }
});


window.onload = function() {
  updateVideoSource();
};

