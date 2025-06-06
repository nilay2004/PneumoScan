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

4. Add the model file:
   - Place your trained model file (`vgg16_model_new.keras`) in the `models` directory
   - See `models/README.md` for more details

5. Run the application:
```bash
python app.py
```

6. Open your browser and navigate to `http://localhost:5000`

## Deployment

### Deploying to Render

1. Create a Render account at https://render.com

2. Create a new Web Service:
   - Connect your GitHub repository
   - Name: `pneumoscan` (or your preferred name)
   - Environment: `Python 3`
   - Build Command: `./build.sh`
   - Start Command: `gunicorn app:app`

3. Add Environment Variables:
   - `PYTHON_VERSION`: `3.11.0`
   - `FLASK_APP`: `app.py`
   - `FLASK_ENV`: `production`

4. Add the Model File:
   - Before deploying, make sure to add your model file to the `models` directory
   - The file should be named `vgg16_model_new.keras`
   - Commit and push the model file to your repository

5. Deploy:
   - Click "Create Web Service"
   - Wait for the deployment to complete
   - Your application will be available at `https://your-app-name.onrender.com`

## Project Structure

```
PneumoScan/
├── app.py              # Main Flask application
├── models/            # Directory for model files
│   └── README.md     # Model setup instructions
├── static/            # Static files (CSS, JS, images)
├── templates/         # HTML templates
├── requirements.txt   # Python dependencies
├── build.sh          # Build script for deployment
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