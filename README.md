# SMS Spam Classification Project

## Overview
This project refactors an SMS text classification solution into a function that constructs a **Linear Support Vector Classification (SVC)** model. Once the model is trained, a **Gradio app** is created to host the application, allowing users to test text messages. The app provides feedback to users, indicating whether the text is classified as **spam** or **not spam** based on the model's predictions.

The project demonstrates how to:
1. Preprocess and split SMS text data.
2. Build and train a machine learning pipeline using **TF-IDF Vectorization** and **LinearSVC**.
3. Deploy the trained model as an interactive web application using **Gradio**.

---

## Project Structure
The project consists of two main components:
1. **SMS Classification Function**:
   - Reads the `SMSSpamCollection.csv` dataset.
   - Splits the data into training and testing sets.
   - Builds a machine learning pipeline using `TfidfVectorizer` and `LinearSVC`.
   - Trains the model on the training data and returns the fitted pipeline.

2. **Gradio App**:
   - Hosts the trained model as an interactive web application.
   - Allows users to input an SMS message and receive a classification result (spam or not spam).
   - Provides a user-friendly interface with clear input and output labels.

---

## Key Features
- **Data Preprocessing**:
  - The dataset is preprocessed to extract features (`text_message`) and labels (`label`).
  - The data is split into training and testing sets (67% training, 33% testing).

- **Machine Learning Pipeline**:
  - Uses **TF-IDF Vectorization** to convert text messages into numerical features.
  - Employs **LinearSVC** for classification, ensuring efficient and accurate predictions.

- **Gradio Interface**:
  - Provides a simple and intuitive interface for users to interact with the model.
  - Displays the classification result in real-time.

---

## How to Use
1. **Train the Model**:
   - The `sms_classification` function reads the dataset, preprocesses the data, and trains the model.
   - The trained model is returned as a fitted pipeline.

2. **Run the Gradio App**:
   - The Gradio app is launched using the `app.launch()` method.
   - Users can input an SMS message into the textbox and click "Submit" to see the classification result.

3. **Interpret the Results**:
   - The app displays whether the input message is classified as **spam** or **not spam**.

---

## Requirements
To run this project, you need the following Python libraries:
- `pandas`
- `scikit-learn`
- `gradio`

You can install the required libraries using:
```bash
pip install pandas scikit-learn gradio
```
---

## Dataset
The dataset used in this project is the **SMSSpamCollection.csv**, which contains SMS messages labeled as either `spam` or `ham` (not spam). The dataset is preprocessed and split into training and testing sets for model training and evaluation.

---

## Future Improvements
- Add model evaluation metrics (e.g., accuracy, precision, recall) to assess performance.
- Deploy the Gradio app to a cloud platform for public access.
- Improve the model by experimenting with different algorithms or hyperparameters.
---

## Conclusion
This project successfully demonstrates the end-to-end process of building and deploying an SMS spam classification model. By leveraging **TF-IDF Vectorization**and **LinearSVC**, the model achieves accurate predictions, while the **Gradio app** provides an intuitive interface for users to interact with the model. This project serves as a foundation for further improvements, such as enhancing model performance, adding evaluation metrics, and deploying the app for broader accessibility. It highlights the power of combining machine learning with user-friendly tools to create practical and impactful solutions.