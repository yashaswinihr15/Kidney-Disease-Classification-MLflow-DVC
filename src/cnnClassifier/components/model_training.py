import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:

    def __init__(self, config: TrainingConfig):
        self.config = config

    # ==========================================================
    # LOAD BASE MODEL + FINE-TUNING
    # ==========================================================

    def get_base_model(self):

        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

        # ------------------------------------------------------
        # Freeze all layers first
        # ------------------------------------------------------

        for layer in self.model.layers:
            layer.trainable = False

        # ------------------------------------------------------
        # Fine-tune the last 6 layers
        # ------------------------------------------------------

        for layer in self.model.layers[-6:]:
            layer.trainable = True

        # ------------------------------------------------------
        # Compile model
        # ------------------------------------------------------

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-5
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        print("\n========================================")
        print("FINE-TUNING CONFIGURATION")
        print("========================================")

        trainable_params = sum(
            tf.keras.backend.count_params(w)
            for w in self.model.trainable_weights
        )

        non_trainable_params = sum(
            tf.keras.backend.count_params(w)
            for w in self.model.non_trainable_weights
        )

        print(
            "Trainable parameters:",
            trainable_params
        )

        print(
            "Non-trainable parameters:",
            non_trainable_params
        )

        print("========================================\n")

    # ==========================================================
    # TRAIN / VALIDATION DATA GENERATORS
    # ==========================================================

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale=1.0 / 255.0,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="categorical"
        )

        # ------------------------------------------------------
        # Validation generator
        # ------------------------------------------------------

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # ------------------------------------------------------
        # Training generator
        # ------------------------------------------------------

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

        # ------------------------------------------------------
        # IMPORTANT: Check class mapping
        # ------------------------------------------------------

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

        # ------------------------------------------------------
        # Count images per class
        # ------------------------------------------------------

        print("\nImages per class:")

        for class_name, class_index in self.train_generator.class_indices.items():

            count = self.train_generator.classes.tolist().count(
                class_index
            )

            print(
                f"  {class_name}: {count} training images"
            )

        print("========================================\n")

    # ==========================================================
    # SAVE MODEL
    # ==========================================================

    @staticmethod
    def save_model(
        path: Path,
        model: tf.keras.Model
    ):

        model.save(path)

    # ==========================================================
    # TRAIN MODEL
    # ==========================================================

    def train(self):

        import math
        import numpy as np

        from sklearn.utils.class_weight import compute_class_weight

        from tensorflow.keras.callbacks import (
            EarlyStopping,
            ReduceLROnPlateau
        )

        # ------------------------------------------------------
        # Calculate class weights
        # ------------------------------------------------------

        classes = np.unique(
            self.train_generator.classes
        )

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=self.train_generator.classes
        )

        class_weights = dict(
            zip(
                classes,
                class_weights
            )
        )

        print("\n========================================")
        print("CLASS WEIGHTS")
        print("========================================")

        print(class_weights)

        print("========================================\n")

        # ------------------------------------------------------
        # Calculate training steps
        # ------------------------------------------------------

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

        # ------------------------------------------------------
        # Early stopping
        # ------------------------------------------------------

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1
        )

        # ------------------------------------------------------
        # Reduce learning rate
        # ------------------------------------------------------

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=1,
            min_lr=1e-7,
            verbose=1
        )

        # ------------------------------------------------------
        # TRAIN
        # ------------------------------------------------------

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.valid_generator,
            validation_steps=self.validation_steps,
            class_weight=class_weights,
            callbacks=[
                early_stopping,
                reduce_lr
            ]
        )

        # ------------------------------------------------------
        # SAVE MODEL
        # ------------------------------------------------------

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

        print(
            f"\nModel saved successfully at: "
            f"{self.config.trained_model_path}"
        )