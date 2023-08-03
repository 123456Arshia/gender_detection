# gender_detection
This repository contains a gender detection model implemented using deep learning techniques. The model is designed to predict the gender of individuals based on their images. It uses a Convolutional Neural Network (CNN) architecture to automatically extract features from the input images and make accurate gender predictions.





# Gender Detection Convolutional Neural Network (CNN)

This repository contains an implementation of a Convolutional Neural Network (CNN) for gender detection using images. The CNN is built with TensorFlow and Keras, and it can predict the gender of a person as either "male" or "female" based on an input image.

## Dataset

The dataset used for training and testing the model can be obtained from https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset. The dataset consists of images of individuals, categorized into two classes: "male" and "female". The data is organized in a directory structure where each class has its own folder containing the corresponding images.

## Requirements

To run this code, you need the following libraries:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- OpenCV (cv2)

You can install the required libraries using the following command:

```bash
pip install tensorflow keras opencv-python numpy
```

## Usage

1. Clone this repository to your local machine:

```bash
git clone https://github.com/123456Arshia/gender-detection-cnn.git
```

2. Download the dataset and place it in the `archive` directory.

3. Open the `gender_detection_cnn.py` file and set the `data_directory` variable to the path of the dataset on your machine.

4. Run the script:

```bash
python gender_detection_cnn.py
```

5. The model will be trained and evaluated using the dataset. After training, the model will be saved as `gender_detector_model.h5` in the same directory.

6. You can now use the trained model to make gender predictions on new images.

## Making Predictions

To make predictions on a new image, follow these steps:

1. Load the trained model using `load_model`:

```python
from tensorflow.keras.models import load_model

model = load_model('gender_detector_model.h5')
```

2. Load and preprocess the new image:

```python
import cv2
import numpy as np

img_path = 'path_to_your_image.jpg'
img_height, img_width = 150, 150

img = cv2.imread(img_path)
img = cv2.resize(img, (img_height, img_width))
img = img / 255.0
img = np.expand_dims(img, axis=0)
```

3. Make the gender prediction:

```python
prediction = model.predict(img)
gender_label = 'female' if prediction[0][0] < 0.5 else 'male'
print(f"Gender prediction: {gender_label}")
```

## Customization

You can customize the CNN architecture, training parameters, and other hyperparameters in the `gender_detection_cnn.py` file to improve the model's performance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

The dataset used in this project was collected from https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset. We gratefully acknowledge the authors and contributors to the dataset.

## Contribution

Contributions to this repository are welcome. If you encounter any issues or have suggestions for improvement, please open an issue or create a pull request.

**Happy coding!**
