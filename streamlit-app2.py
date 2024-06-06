with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        with st.spinner('Calculating...'):
            # Update text to indicate the progress
            st.text("Model 1: Calculating SHAP values...")
            explainer1 = shap.Explainer(pipe1)
            shap_values1 = explainer1([text])  # Pass text directly as a list
            st.text("Model 1: SHAP values calculated.")

            # Update text to indicate the progress
            st.text("Model 2: Calculating SHAP values...")
            explainer2 = shap.Explainer(pipe2)
            shap_values2 = explainer2([text])  # Pass text directly as a list
            st.text("Model 2: SHAP values calculated.")

        if selected_model1 == "nlptown/bert-base-multilingual-uncased-sentiment":
            st.text("Negative: Negative sentiment, Neutral: Neutral sentiment, Positive: Positive sentiment")
            st.text("Negative")
            st_shap(shap.plots.text(shap_values1[:, :, "Negative"]))
            st.text("Neutral")
            st_shap(shap.plots.text(shap_values1[:, :, "Neutral"]))
            st.text("Positive")
            st_shap(shap.plots.text(shap_values1[:, :, "Positive"]))
        elif selected_model1 == "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis":
            st.text("Negative: Negative sentiment, Neutral: Neutral sentiment, Positive: Positive sentiment")
            st.text("Negative")
            st_shap(shap.plots.text(shap_values1[:, :, "negative"]))
            st.text("Neutral")
            st_shap(shap.plots.text(shap_values1[:, :, "neutral"]))
            st.text("Positive")
            st_shap(shap.plots.text(shap_values1[:, :, "positive"]))
        elif selected_model1 == "ElKulako/cryptobert":
            st.text("Bullish: Positive sentiment, Neutral: Neutral sentiment, Bearish: Negative sentiment")
            st.text("Bullish")
            st_shap(shap.plots.text(shap_values1[:, :, "Bullish"]))
            st.text("Neutral")
            st_shap(shap.plots.text(shap_values1[:, :, "Neutral"]))
            st.text("Bearish")
            st_shap(shap.plots.text(shap_values1[:, :, "Bearish"]))

        if selected_model2 == "nlptown/bert-base-multilingual-uncased-sentiment":
            st.text("Negative: Negative sentiment, Neutral: Neutral sentiment, Positive: Positive sentiment")
            st.text("Negative")
            st_shap(shap.plots.text(shap_values2[:, :, "Negative"]))
            st.text("Neutral")
            st_shap(shap.plots.text(shap_values2[:, :, "Neutral"]))
            st.text("Positive")
            st_shap(shap.plots.text(shap_values2[:, :, "Positive"]))
        elif selected_model2 == "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis":
            st.text("Negative: Negative sentiment, Neutral: Neutral sentiment, Positive: Positive sentiment")
            st.text("Negative")
            st_shap(shap.plots.text(shap_values2[:, :, "negative"]))
            st.text("Neutral")
            st_shap(shap.plots.text(shap_values2[:, :, "neutral"]))
            st.text("Positive")
            st_shap(shap.plots.text(shap_values2[:, :, "positive"]))
        elif selected_model2 == "ElKulako/cryptobert":
            st.text("Bullish: Positive sentiment, Neutral: Neutral sentiment, Bearish: Negative sentiment")
            st.text("Bullish")
            st_shap(shap.plots.text(shap_values2[:, :, "Bullish"]))
            st.text("Neutral")
            st_shap(shap.plots.text(shap_values2[:, :, "Neutral"]))
            st.text("Bearish")
            st_shap(shap.plots.text(shap_values2[:, :, "Bearish"]))