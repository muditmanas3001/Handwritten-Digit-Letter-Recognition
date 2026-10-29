Handwritten Character Recognition Using CNNs and PyTorch
-
A handwritten uppercase letter recognition system built with a custom Convolutional Neural Network (CNN) in PyTorch. It uses the EMNIST Letters dataset and a Flask web app where users can draw a character and view the model’s top three predictions.

Table of Contents
-
- Overview
- Objectives
- Existing Approaches
- Dataset
- Methodology
- Model Architecture
- Web Application
- Challenges and Observations
- Conclusion
- Future Work
- References
  
Overview
-
Handwriting varies widely in shape, stroke width, spacing, and orientation, which makes automatic recognition challenging. Traditional OCR methods often rely on hand-crafted features and can struggle to generalize across writing styles.

This project explores a deep learning approach: a CNN learns visual features directly from handwritten character images. The trained model is served through a Flask application that accepts drawings from a browser canvas and returns predicted letters with confidence scores.

Objectives
-
- Train a CNN to classify handwritten uppercase English letters.
- Use PyTorch and the EMNIST Letters dataset for model development.
- Build a web interface for drawing and submitting a character.
- Return the three most likely predictions to make uncertain results easier to interpret.
- Investigate how preprocessing and input orientation affect recognition.
- 
Existing Approaches
-
Template matching
Template matching compares an input image with stored examples. It is simple, but sensitive to changes in scale, rotation, and writing style.
Statistical methods and Hidden Markov Models
Statistical classifiers and Hidden Markov Models have been used to model handwriting features and stroke sequences. These methods often depend on carefully designed feature extraction and can be difficult to generalize across varied handwriting.
SVMs with HOG features
Support Vector Machines trained on Histogram of Oriented Gradients features can provide effective baselines. Their performance depends on the selected features and preprocessing pipeline.
Convolutional Neural Networks
CNNs learn spatial features from image pixels, progressing from basic edges and curves to patterns useful for distinguishing character classes. This reduces the need for manual feature engineering.
Dataset
The project uses the Letters split of the EMNIST (Extended MNIST) dataset, which contains 145,600 28 × 28 grayscale images across 26 letter classes.

The dataset provides 124,800 training examples and 20,800 test examples. For this project, 90% of the combined data was selected to reduce computational cost, then divided into training, validation, and test subsets in an 80:10:10 ratio. A random seed of 42 was used for reproducibility.

This split corresponds to approximately 72%, 9%, and 9% of the full combined dataset, respectively; the remaining 10% is excluded. The exact sample counts depend on the split implementation and rounding.
Preprocessing
- Convert images to grayscale with one input channel.
- Resize images to 28 × 28 pixels.
- Normalize pixel values using mean 0.5 and standard deviation 0.5, mapping values approximately to [-1, 1].
- Prepare browser canvas input to match the orientation, polarity, size, and value range expected by the model.
  
Methodology
-
The system follows this workflow:
1. Load and preprocess the EMNIST Letters images.
2. Split the selected data into training, validation, and test subsets.
3. Train a CNN using PyTorch.
4. Preprocess a character drawn in the browser.
5. Run inference and display the top three predicted classes and their confidence scores.
The canvas preprocessing described for this project includes image inversion, horizontal flipping, and a 90-degree rotation to align user drawings with the orientation expected by the model. The exact transformations should be validated against the dataset and training pipeline.
Model Architecture
The project uses a lightweight CNN named SimpleCNN.
Layer	Configuration
Input	1 × 28 × 28 grayscale image
Convolution	1 input channel, 16 output channels, 3 × 3 kernel, stride 1, padding 2
Activation and pooling	ReLU followed by 2 × 2 max pooling
Second pooling	2 × 2 max pooling
Flatten	16 × 7 × 7 = 784 features
Fully connected 1	784 → 256, ReLU
Fully connected 2	256 → 64, ReLU
Output	64 → 26 letter classes, or 27 if an additional class is explicitly defined and trained


With a 3 × 3 convolution and padding 2, the convolution output is 30 × 30. After the first 2 × 2 max-pooling operation it is 15 × 15; the second pooling operation reduces it to 7 × 7.

The convolutional layer has 160 trainable parameters. With 26 output classes, the architecture described above has 219,258 trainable parameters. A 27-class output has 219,323 parameters, but the extra class must have a defined label and training examples; it should not be described as a space class unless the dataset and label mapping include one.
Web Application
A Flask web application provides a browser-based drawing canvas. Users can draw a character and submit it for prediction. The application displays the top three predicted letters and their confidence probabilities.

The web app is intended as an interactive demonstration of the trained model. Prediction confidence should be treated as a model score, not as a guarantee that a prediction is correct.

Challenges and Observations
-
Limitations of conventional approaches
- Template matching and rule-based methods can be sensitive to handwriting style, pen width, and slant.
- Feature-based approaches such as HOG with SVMs require decisions about feature extraction and preprocessing.
- Systems designed primarily for printed text may not perform reliably on freehand input.
- Differences between training images and live canvas input, including orientation, background, scale, and pixel polarity, can reduce accuracy.
- Batch evaluation alone does not demonstrate how a model behaves in an interactive application.
How this project addresses them
- The CNN learns features directly from image pixels rather than relying entirely on hand-crafted features.
- Canvas preprocessing is used to make browser input more consistent with the training data.
- The Flask interface supports interactive, real-time inference.
- The top-three result display gives users alternatives when the model is uncertain.
- PyTorch makes it practical to adjust the architecture and preprocessing during development.

  
Conclusion
-
This project demonstrates an end-to-end handwritten uppercase character recognition workflow using PyTorch, EMNIST Letters, and Flask. It combines model training, input preprocessing, and interactive browser-based inference in one application.

A key implementation lesson is that training and inference preprocessing must agree. Orientation, image polarity, scaling, and normalization can materially affect predictions. Quantitative performance should be reported using measured validation and test metrics rather than qualitative impressions alone.

Future Work
-
- Evaluate deeper architectures such as ResNet-18 or MobileNetV2.
- Add data augmentation such as small rotations, shearing, and elastic distortions.
- Expand recognition to digits and additional letter classes with a suitable EMNIST split.
- Add word-level recognition by combining character predictions with a language model.
- Explore mobile deployment using TorchScript or ONNX.
- Add a user feedback workflow for collecting corrections and improving the model.
- Investigate Vision Transformer architectures for character classification.


References
-
1. G. Cohen, S. Afshar, J. Tapson, and A. van Schaik, “EMNIST: Extending MNIST to handwritten letters,” Proceedings of the International Joint Conference on Neural Networks (IJCNN), 2017, pp. 2921–2926.
2. Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, November 1998.
3. EMNIST dataset
4. PyTorch documentation
5. Flask documentation
