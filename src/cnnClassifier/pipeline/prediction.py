import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):

        # Load trained model
        model = load_model(
            os.path.join("artifacts", "training", "model.h5")
        )

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
        test_image = np.expand_dims(test_image, axis=0)

        # Prediction
        predictions = model.predict(test_image)

        print("Prediction probabilities:", predictions)

        result = np.argmax(predictions, axis=1)

        print("Predicted class index:", result[0])

        # Current dataset mapping
        class_names = ["Normal", "Tumor"]

        prediction = class_names[result[0]]

        return [{"image": prediction}]