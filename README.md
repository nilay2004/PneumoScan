# PneumoScan - AI-Powered Pneumonia Detection

PneumoScan is an advanced web-based application that leverages artificial intelligence to detect pneumonia from chest X-ray images. Using a deep learning model based on the VGG16 architecture, it achieves high accuracy in identifying pneumonia patterns in chest radiographs.

## Features

- **Instant Analysis**: Upload chest X-ray images and receive results within seconds
- **High Accuracy**: Trained on a comprehensive dataset
- **User-Friendly Interface**: Modern, intuitive web interface for easy interaction
- **Secure Processing**: Ensures patient data privacy and security
- **Responsive Design**: Accessible across all devices and screen sizes

## Technology Stack

### Machine Learning
- VGG16 Architecture
- TensorFlow 2.15.0
- Keras 2.15.0

### Web Development
- Flask 3.0.3
- HTML/CSS
- JavaScript

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nilay2004/PneumoScan.git
cd PneumoScan
```

2. Create and activate a virtual environment:
```bash
python -m venv env
# On Windows:
env\Scripts\activate
# On Unix or MacOS:
source env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
PneumoScan/
├── app.py              # Main Flask application
├── models/            # Directory for model files (not included in repo)
├── static/            # Static files (CSS, JS, images)
├── templates/         # HTML templates
├── requirements.txt   # Python dependencies
└── README.md         # Project documentation
```

## Usage

1. Navigate to the home page
2. Click "Try It Now" or upload button
3. Upload a chest X-ray image (supported formats: JPEG, PNG)
4. Wait for the analysis to complete
5. View the results and recommended actions

## Model Information

The application uses a VGG16-based model for pneumonia detection. The model is trained on chest X-ray images and can classify them as either normal or pneumonia cases.

Note: The model files are not included in this repository due to their size. You'll need to train the model using the provided training scripts or obtain the model files separately.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Nilay Pandya - Lead Developer
- GitHub: [@nilay2004](https://github.com/nilay2004)

## Acknowledgments

- Chest X-ray Pneumonia Dataset
- VGG16 model architecture
- Flask framework and its contributors 