document.addEventListener('DOMContentLoaded', (event) => {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureButton = document.getElementById('captureButton');
    const submitButton = document.getElementById('submitButton');
    const imageData = document.getElementById('imageData');

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(err => {
            console.error("Error accessing the camera", err);
        });

    captureButton.addEventListener('click', () => {
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
        imageData.value = canvas.toDataURL('image/jpeg');
        submitButton.disabled = false;
        captureButton.textContent = 'Recapture';
    });

    document.getElementById('captureForm').addEventListener('submit', (e) => {
        if (!imageData.value) {
            e.preventDefault();
            alert('Please capture an image before submitting.');
        }
    });

    document.getElementById('name').addEventListener('input', function() {
        submitButton.disabled = !this.value || !imageData.value;
    });
});