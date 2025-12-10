import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

RUN_FULL_TRAIN = True
EARLY_STOPPING_PATIENCE = 5
NUM_EPOCHS = 30

class DataImporter:
    def __init__(self):
        self.DATASET_SLUG = "drgfreeman/rockpaperscissors"
        self.DATASET_URL = f"https://www.kaggle.com/datasets/{self.DATASET_SLUG}"

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(script_dir, "data")

    def import_data(self):
        print("Dataset URL:", self.DATASET_URL)
        print("DATA_DIR:", self.DATA_DIR)

        os.makedirs(self.DATA_DIR, exist_ok=True)

        def find_dataset_folder():
            for root, dirs, files in os.walk(self.DATA_DIR):
                dir_names = set(dirs)
                if {"rock", "paper", "scissors"}.issubset(dir_names):
                    return root
            return None

        dataset_folder = find_dataset_folder()

        if dataset_folder is None:
            print("Dataset not found locally. Downloading from Kaggle...")

            api = KaggleApi()
            api.authenticate()

            api.dataset_download_files(
                self.DATASET_SLUG,
                path=self.DATA_DIR,
                unzip=True
            )

            print("Download and extract completed.")
            dataset_folder = find_dataset_folder()
            if dataset_folder is None:
                raise RuntimeError(
                    f"Download finished, but could not find 'rock','paper','scissors' under {self.DATA_DIR}"
                )
        else:
            print("Dataset already exists locally:", dataset_folder)

        print("USING DATASET FOLDER:", dataset_folder)
        return dataset_folder

importer = DataImporter()
data_dir = importer.import_data()

try:
    import tensorflow as tf
    from tensorflow.keras.utils import image_dataset_from_directory
    
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(150, 150),
        batch_size=32,
    )
    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(150, 150),
        batch_size=32,
    )
    class_names = train_ds.class_names
    print('Classes:', class_names)
    for images, labels in train_ds.take(1):
        print('Batch shape:', images.shape, 'Labels shape:', labels.shape)
    try:
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

        model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255, input_shape=(150,150,3)),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint('best_rps_model.keras', save_best_only=True, monitor='val_loss'),
        ]
        if not RUN_FULL_TRAIN:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=EARLY_STOPPING_PATIENCE,
                    restore_best_weights=True
                )
            )
        print(f"Training config: RUN_FULL_TRAIN={RUN_FULL_TRAIN}, EARLY_STOPPING_PATIENCE={EARLY_STOPPING_PATIENCE}, NUM_EPOCHS={NUM_EPOCHS}")
        print('Callbacks:', [type(cb).__name__ for cb in callbacks])

        history = model.fit(train_ds, validation_data=val_ds, epochs=NUM_EPOCHS, callbacks=callbacks)

        model.save('rps_cnn_final.keras')

        test_loss, test_acc = model.evaluate(val_ds)
        print('Final test accuracy:', test_acc)

        import numpy as np
        import matplotlib.pyplot as plt

        val_images = np.concatenate([x.numpy() for x, y in val_ds], axis=0)
        val_labels = np.concatenate([y.numpy() for x, y in val_ds], axis=0)

        preds = model.predict(val_images)
        pred_labels = np.argmax(preds, axis=1)

        mis_idx = np.where(pred_labels != val_labels)[0]
        print('Number of misclassified examples in validation set:', len(mis_idx))

        n_show = min(6, len(mis_idx))
        for i in range(n_show):
            idx = mis_idx[i]
            plt.figure(figsize=(3,3))
            plt.imshow(val_images[idx].astype('uint8'))
            plt.title(f"True: {class_names[pred_labels[idx]]}  Pred: {class_names[val_labels[idx]]}")
            plt.axis('off')
            plt.show()

        print('\nPossible reasons for confusion: small inter-class variations, similar hand poses, different lighting/backgrounds, or low image resolution.')
    except Exception as e:
        print('Model build/train/eval skipped or failed:', e)
except Exception as e:
    print('Could not load TensorFlow or create datasets:', e)