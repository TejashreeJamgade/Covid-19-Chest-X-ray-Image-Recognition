# Covid-19-Chest-X-Ray-Image-Recognition

**Covid-19 Chest XRay Image Recognition**

**Introduction:**

The Covid-19 Chest XRay Image Recognition project aims to build an image classification model that can identify Covid-19-positive patients by analyzing their chest X-ray images. The project utilizes computer vision techniques to distinguish between healthy and Covid-19 infected lungs based on visual patterns present in the X-ray images. This project has significant real-world applications in healthcare, especially during the ongoing Covid-19 pandemic.

**Dataset:**

The dataset used for this project is the "Covid-19 Image Dataset" by Pranav Raikokte, available on Kaggle. The dataset contains chest X-ray images of Covid-19 positive and negative patients, as well as patients with other lung diseases. It comprises three classes: Covid-19 Positive, Covid-19 Negative, and Virus (other lung diseases).

Dataset Source: [Kaggle Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)

**Model Building and Evaluation:**

The image classification model was built using a sequential model architecture in Keras. The model consists of three convolutional layers with ReLU activation functions and max-pooling layers to extract features from the images. Dropout layers were added to reduce overfitting. The last layers include a flattened layer, a dense layer with a ReLU activation, another dropout layer, and the final output layer with a softmax activation function to predict the probability of each class.

The model was compiled using the categorical cross-entropy loss function, Adam optimizer, and accuracy as the evaluation metric. It was trained on the preprocessed dataset using data augmentation techniques for 20 epochs with a batch size of 16.

**Results and Discussion:**

The model achieved an accuracy of X% on the test dataset. The precision, recall, and F1-score for each class were as follows:
- Covid-19 Positive: Precision
- Covid-19 Negative: Precision 
- Virus: Precision

The confusion matrix showed that the model performed well in correctly classifying Covid-19 positive and negative cases. However, there were some misclassifications between Covid-19 negative and virus cases, indicating areas for potential improvement.

**Future Work:**

Possible areas of future improvement include:
- Fine-tuning the model's architecture to achieve better performance.
- Exploring other deep learning architectures or transfer learning techniques.
- Collecting a larger and more diverse dataset to further improve model generalization.
- Investigating ways to handle class imbalances if present in the dataset.

**Getting Started:**

1. Clone the repository:
```
git clone https://github.com/Asadxio/Covid19-ChestXRay-Recognition.git
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Download the dataset from [Kaggle Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset) and place it in the appropriate directory.

4. Run the Jupyter Notebook or Python script to train the model and make predictions.

**Contributing:**

Contributions to this project are welcome! If you find any issues or want to add new features, feel free to open a pull request.

**License:**

This project is shared under the [MIT License](https://opensource.org/licenses/MIT).

**Author:**

- Asad Ali
- Contact: aliasadcivil007@gmail.com
- LinkedIn: [ LinkedIn Profile](https://www.linkedin.com/in/asad-ali-mulla-044240262/))

**Acknowledgments:**

Special thanks to Pranav Raikokte for providing the Covid19 Image Dataset on Kaggle.
