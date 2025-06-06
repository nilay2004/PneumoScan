function uploadImage() {
    const fileInput = document.getElementById('xrayImage');
    const resultDiv = document.getElementById('result');
    const uploadedImage = document.getElementById('uploadedImage');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const predictBtn = document.getElementById('predictBtn');

    if (!fileInput.files[0]) {
        resultDiv.innerHTML = 'Please select an image.';
        return;
    }

    // Validate file type
    const file = fileInput.files[0];
    if (!file.type.match('image.*')) {
        resultDiv.innerHTML = 'Please select a valid image file.';
        return;
    }

    // Show loading indicator
    loadingIndicator.classList.remove('hidden');
    resultDiv.classList.add('hidden');
    predictBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            resultDiv.innerHTML = data.error;
        } else {
            resultDiv.innerHTML = `Prediction: ${data.result}`;
            uploadedImage.src = data.image_path;
            uploadedImage.style.display = 'block';
        }
    })
    .catch(error => {
        resultDiv.innerHTML = 'Error processing image. Please try again.';
        console.error('Error:', error);
    })
    .finally(() => {
        loadingIndicator.classList.add('hidden');
        resultDiv.classList.remove('hidden');
        predictBtn.disabled = false;
    });
}

function refreshComparison() {
    const img = document.getElementById('model-comparison');
    const src = img.src;
    img.src = '';
    img.src = src + '?t=' + new Date().getTime(); // Force reload
}

function updateActiveNav() {
    const navLinks = document.querySelectorAll('.nav-links a');
    const currentPath = window.location.pathname;

    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

function initializeComparisonTable() {
    const tableRows = document.querySelectorAll('.comparison-section table tbody tr');
    tableRows.forEach(row => {
        row.addEventListener('mouseover', () => {
            row.style.backgroundColor = '#f0f4ff';
        });
        row.addEventListener('mouseout', () => {
            if (!row.classList.contains('bg-blue-50')) {
                row.style.backgroundColor = '';
            }
        });
    });
}

document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('xrayImage');
    const uploadedImage = document.getElementById('uploadedImage');
    const predictBtn = document.getElementById('predictBtn');
    const browseText = document.querySelector('.browse-text');
    const previewContainer = document.querySelector('.preview-container');
    const resultDiv = document.getElementById('result');
    const loadingIndicator = document.getElementById('loadingIndicator');

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('drag-over');
    }

    function unhighlight(e) {
        dropZone.classList.remove('drag-over');
    }

    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length) {
            fileInput.files = files;
            handleFileSelect();
        }
    }

    // Handle file selection via browse button
    browseText.addEventListener('click', () => {
        fileInput.click();
    });

    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', handleFileSelect);

    function handleFileSelect() {
        if (fileInput.files.length) {
            const file = fileInput.files[0];
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    uploadedImage.src = e.target.result;
                    previewContainer.classList.add('active');
                    predictBtn.disabled = false;
                    resultDiv.innerHTML = '';
                };
                reader.readAsDataURL(file);
            } else {
                resultDiv.innerHTML = 'Please select a valid image file.';
                previewContainer.classList.remove('active');
                predictBtn.disabled = true;
            }
        }
    }

    // Handle prediction button click
    predictBtn.addEventListener('click', function() {
        if (!fileInput.files.length) {
            resultDiv.innerHTML = 'Please select an image first.';
            return;
        }

        loadingIndicator.classList.remove('hidden');
        resultDiv.classList.add('hidden');
        predictBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                resultDiv.innerHTML = data.error;
            } else {
                resultDiv.innerHTML = `Prediction: ${data.result}`;
                uploadedImage.src = data.image_path;
                uploadedImage.style.display = 'block';
            }
        })
        .catch(error => {
            resultDiv.innerHTML = 'Error processing image. Please try again.';
            console.error('Error:', error);
        })
        .finally(() => {
            loadingIndicator.classList.add('hidden');
            resultDiv.classList.remove('hidden');
            predictBtn.disabled = false;
        });
    });
});