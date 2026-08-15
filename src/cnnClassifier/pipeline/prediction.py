import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class PredictionPipeline:

    def __init__(self, filename):

        self.filename = filename

        # Load model ONLY ONCE
        model_path = os.path.join(
            "artifacts",
            "training",
            "model.h5"
        )

        print("Loading model...")
        self.model = load_model(model_path)
        print("Model loaded successfully!")

        # IMPORTANT:
        # This mapping matches your training output:
        # Normal = 0
        # Tumor  = 1
        self.class_names = ["Normal", "Tumor"]


    def predict(self):

        # Load image
        test_image = image.load_img(
            self.filename,
            target_size=(224, 224)
        )

        # Convert image to array
        test_image = image.img_to_array(test_image)

        # Same preprocessing used during training
        test_image = test_image / 255.0

        # Add batch dimension
        test_image = np.expand_dims(
            test_image,
            axis=0
        )

        # Prediction
        predictions = self.model.predict(
            test_image,
            verbose=0
        )

        print(
            "Prediction probabilities:",
            predictions
        )

        # Get predicted class
        predicted_index = np.argmax(
            predictions,
            axis=1
        )[0]

        print(
            "Predicted class index:",
            predicted_index
        )

        prediction = self.class_names[predicted_index]

        print(
            "Final prediction:",
            prediction
        )

        return [
            {
                "image": prediction
            }
        ]