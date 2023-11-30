# MonReader (Computer Vision)

# Background:

Our company develops innovative Artificial Intelligence and Computer Vision solutions that revolutionize industries. Machines that can see: We pack our solutions in small yet intelligent devices that can be easily integrated to your existing data flow. Computer vision for everyone: Our devices can recognize faces, estimate age and gender, classify clothing types and colors, identify everyday objects and detect motion. Technical consultancy: We help you identify use cases of artificial intelligence and computer vision in your industry. Artificial intelligence is the technology of today, not the future.

![image](https://github.com/kuzhuppillil/MonReader/assets/25860818/a97f71f8-2b44-4b74-86a0-a3b53372b5af)

MonReader is a new mobile document digitalization experience for the blind, for researchers and for everyone else in need for fully automatic, highly fast and high-quality document scanning in bulk. It is composed of a mobile app and all the user needs to do is flip pages and everything is handled by MonReader: it detects page flips from low-resolution camera preview and takes a high-resolution picture of the document, recognizing its corners and crops it accordingly, and it dewarps the cropped document to obtain a bird's eye view, sharpens the contrast between the text and the background and finally recognizes the text with formatting kept intact, being further corrected by MonReader's ML powered redactor.

![image](https://github.com/kuzhuppillil/MonReader/assets/25860818/0d86a22d-ca67-41dd-8647-29153ec3384e)


# Data Description:
We collected page flipping video from smart phones and labelled them as flipping and not flipping.
We clipped the videos as short videos and labelled them as flipping or not flipping. The extracted frames are then saved to disk in a sequential order with the following naming structure: VideoID_FrameNumber

    


# Goal(s):
* Predict if the page is being flipped using a single image.
* Success Metrics: Evaluate model performance based on F1 score, the higher the better.
* Bonus(es): Predict if a given sequence of images contains an action of flipping. 

# Solution:
The dataset comprises sequentially ordered frames with a naming convention: VideoID_FrameNumber (e.g., 0001_000000010.jpg). These frames are organized into three main directories, namely 'training', 'testing' and 'validation'. Each of these directories contains two subdirectories labeled 'flip' and 'notflip,' categorizing the images accordingly. 

      Total training flip images: 1000
      Total training notflip images: 1000
      Total validation flip images: 162
      Total validation notflip images: 230
      Total testing flip images: 290
      Total testing notflip images: 307

![image](https://github.com/kuzhuppillil/Xwd9tHcXQqtfbaKV/assets/25860818/2b879449-05c4-4471-a099-8f487675b539)

## Data Preprocessing:
  
* The pixel values of the images are normalized, transforming the original range from (0-255) to a normalized range of (0-1).
* Data augmentation is applied to the training images as a strategy to mitigate overfitting. This involves introducing random transformations to each image during the training phase, enhancing the model's ability to generalize to various scenarios


## Building a CNN:
 
* All images are uniformly resized to 150x150 pixels and are in RGB format with three channels.
* Our chosen model configuration is a widely adopted setup, especially suitable for small training datasets. It involves the stacking of three layers: convolution, rectified linear unit (ReLU) activation, and max-pooling. The convolutional layer employs a 3x3 filter, and the max-pooling layer uses a 2x2 filter.
* We utilize 16 filters for the first convolutional layer, followed by 32 filters in the second layer and 64 filters in the third layer. The first filters can capture basic features and edges, the 2nd layers can capture more complex patterns and higher-level features and finally 3rd layer filters is to focus on more abstract features.
* To address overfitting, we introduce dropout, a technique that randomly deactivates some neurons during training to enhance model generalization.
* The final layers of the Convolutional Neural Network (CNN) consist of two fully-connected layers responsible for classification. The output layer employs a sigmoid activation function, ensuring that the output is a single scalar value ranging from 0 to 1.


![image](https://github.com/kuzhuppillil/Xwd9tHcXQqtfbaKV/assets/25860818/c1328925-cdfe-46cc-b026-53660a0cddcf)


* The image below illustrates the intermediate representations of a sample image after it has passed through each layer of the network.

![image](https://github.com/kuzhuppillil/Xwd9tHcXQqtfbaKV/assets/25860818/f49a1f21-9c41-49fb-9736-a44223a78257)


## Final Obervations: 
### Model evaluation:
* The graph below exhibits a consistent rise in accuracy and a steady decline in loss throughout the training process. Simultaneously, the validation metrics closely align with the training metrics, indicating the successful training of our model without encountering overfitting.

![image](https://github.com/kuzhuppillil/Xwd9tHcXQqtfbaKV/assets/25860818/6eb13693-5c98-4ef0-a5fe-8f41c2e2a820)

* Below metrics provide a comprehensive evaluation of the model's ability to correctly classify instances from both classes, with high precision, recall, and F1-score values. The accuracy of 92% suggests an effective overall performance.
  
      notflip class:
            Precision: 96%
            Recall: 87%
            F1-score: 92%
            Support: 300
    
      flip class:
            Precision: 88%
            Recall: 97%
            F1-score: 92%
            Support: 297
    
      Overall Metrics:
            Accuracy: 92%
            Macro Avg (average of precision, recall, and f1-score across classes): 92%
            Weighted Avg (weighted average based on the number of samples in each class): 92%
* According to the confusion matrix, there were misclassifications in the following manner:
  
      For the "notflip" class:
          True Positive (correctly predicted as "notflip"): 300 - 37 = 263
          False Negative (misclassified as "flip"): 37
      For the "flip" class:
          True Positive (correctly predicted as "flip"): 286 - 7 = 279
          False Negative (misclassified as "notflip"): 7


![image](https://github.com/kuzhuppillil/Xwd9tHcXQqtfbaKV/assets/25860818/ab678fc4-ac7f-4593-bc6d-61ef74b08ebd)


### In conclusion, upon reviewing the image sequence classification results below, it is noteworthy that only 1 out of 6 instances was misclassified. This outcome underscores the robustness and effectiveness of our model in accurately classifying image sequences.


![image](https://github.com/kuzhuppillil/Xwd9tHcXQqtfbaKV/assets/25860818/21e9b905-7e97-4939-aad1-0991afdce7fb)


# References:

1. http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/
2. https://keras.io/api/data_loading/image/
3. https://en.wikipedia.org/wiki/Stochastic_gradient_descent#RMSProp
4. https://developers.google.com/machine-learning/practica/image-classification
5. https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a.




