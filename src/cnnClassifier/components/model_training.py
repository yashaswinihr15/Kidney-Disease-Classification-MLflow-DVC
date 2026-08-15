import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:

    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):

        # --------------------------------------------------
        # Image preprocessing
        # --------------------------------------------------
        datagenerator_kwargs = dict(
            rescale=1.0 / 255.0,
            validation_split=0.20
        )

        # --------------------------------------------------
        # Image loading parameters
        # --------------------------------------------------
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="categorical"
        )

        # --------------------------------------------------
        # Validation generator
        # --------------------------------------------------
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # --------------------------------------------------
        # Training generator
        # --------------------------------------------------
        if self.config.params_is_augmentation:

            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )

        else:

            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        # --------------------------------------------------
        # IMPORTANT: Print class mapping
        # --------------------------------------------------
        print("\n========================================")
        print("CLASS INFORMATION")
        print("========================================")

        print(
            "TRAIN CLASS INDICES:",
            self.train_generator.class_indices
        )

        print(
            "VALID CLASS INDICES:",
            self.valid_generator.class_indices
        )

        print(
            "TRAIN SAMPLES:",
            self.train_generator.samples
        )

        print(
            "VALIDATION SAMPLES:",
            self.valid_generator.samples
        )

        print(
            "CLASS NAMES:",
            self.train_generator.class_indices.keys()
        )

        # --------------------------------------------------
        # Print number of images in each class
        # --------------------------------------------------
        print("\nImages per class:")

        for class_name, class_index in self.train_generator.class_indices.items():

            count = self.train_generator.classes.tolist().count(
                class_index
            )

            print(
                f"  {class_name}: {count} training images"
            )

        print("========================================\n")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):

        import math
        import numpy as np
        from sklearn.utils.class_weight import compute_class_weight

        # --------------------------------------------------
        # Calculate class weights
        # --------------------------------------------------
        classes = np.unique(self.train_generator.classes)

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=self.train_generator.classes
        )

        class_weights = dict(
            zip(classes, class_weights)
        )

        print("\n========================================")
        print("CLASS WEIGHTS")
        print("========================================")
        print(class_weights)
        print("========================================\n")

        # --------------------------------------------------
        # Calculate steps
        # --------------------------------------------------
        self.steps_per_epoch = math.ceil(
            self.train_generator.samples /
            self.train_generator.batch_size
        )

        self.validation_steps = math.ceil(
            self.valid_generator.samples /
            self.valid_generator.batch_size
        )

        print(
            "Steps per epoch:",
            self.steps_per_epoch
        )

        print(
            "Validation steps:",
            self.validation_steps
        )

        # --------------------------------------------------
        # Train model
        # --------------------------------------------------
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.valid_generator,
            validation_steps=self.validation_steps,
            class_weight=class_weights
        )

        # --------------------------------------------------
        # Save trained model
        # --------------------------------------------------
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

        print(
            f"\nModel saved successfully at: "
            f"{self.config.trained_model_path}"
        )