# Sentiment Analysis App

That is the app link: https://3crjg5zymizzrj569e9yat.streamlit.app/

This is a Streamlit-based web application for performing sentiment analysis using pre-trained Hugging Face transformer models. The app allows users to input text and analyze its sentiment using various models. It also provides visualizations of the sentiment distribution and SHAP (SHapley Additive exPlanations) values for model interpretability.

## Features

- **Model Selection**: Choose from a list of pre-trained Hugging Face transformer models or specify a custom model.
- **Sentiment Analysis**: Analyze the sentiment of input text using selected models.
- **Sentiment Distribution**: Visualize the sentiment distribution of the input text using bar charts.
- **SHAP Values**: Display SHAP values to explain the model's predictions and understand the importance of input features.

## Installation

1. Clone the repository:

    ```
    git clone https://github.com/your_username/sentiment-analysis-app.git
    ```

2. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:

    ```
    streamlit run app.py
    ```

2. Access the app in your web browser (by default, it should be available at `http://localhost:8501`).

3. Select a model from the dropdown menu or specify a custom model name (if applicable).

4. Enter text in the input field and click the "Analyze" button to perform sentiment analysis.

5. Explore the sentiment distribution visualization and SHAP values to gain insights into the model's predictions.

## Screenshots

[![Screenshot1](screenshots/screenshot1.png)](screenshots/screenshot1.png)

[![Screenshot2](screenshots/screenshot2.png)](screenshots/screenshot2.png)

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
