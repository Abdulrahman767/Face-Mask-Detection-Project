Face Mask Detection Project
This project aims to develop a model that can detect whether a person is wearing a face mask or not. The project involves loading image data, preprocessing it, training a convolutional neural network (CNN), and then using the trained model to make predictions on new images.

**Project Structure

   *Data Preparation

      1-Load images from the with_mask and without_mask directories.
      2-Resize images to 128x128 pixels and convert them to RGB format.
      3-Label the images (1 for with mask, 0 for without mask).
      4-Combine the images and labels into arrays.
      5-Split the data into training and testing sets.

   *Model Training

      1-Define a CNN model using TensorFlow/Keras.
      2-The model consists of convolutional layers, max-pooling layers, dense layers, and dropout layers.
      3-Compile the model using the Adam optimizer and sparse categorical cross-entropy loss function.
      4-Train the model on the training data for 5 epochs.
      5-Evaluate the model on the test data and print the accuracy.

   *Results Visualization

      1-Plot the training and validation loss.
      2-Plot the training and validation accuracy.

   *Prediction

      1-Load an input image specified by the user.
      2-Preprocess the input image.
      3-Use the trained model to predict whether the person in the image is wearing a mask.
      4-Print the prediction result.


**Dependencies

  Python
  NumPy
  Matplotlib
  OpenCV
  PIL (Python Imaging Library)
  Scikit-learn
  TensorFlow
  Keras