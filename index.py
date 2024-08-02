from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

def train(dataset_path, save_path):
    # Data augmentation and preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Load and iterate training dataset
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )

    # Define the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(train_generator.class_indices), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    epochs = 3
    for epoch in range(epochs):
        progress = (epoch + 1) / epochs * 100
        print(f"Training epoch {epoch + 1}/{epochs} - {progress:.2f}% complete")
        model.fit(train_generator, epochs=1)

    # Save the model
    save_name = 'dog-cat'
    file_save_dir = os.path.join(save_path, save_name + '.h5')
    model.save(file_save_dir)

    print(f"Model saved at: {file_save_dir}")


if __name__ == '__main__':
    dataset_path= ''
    save_path=''
    train(dataset_path,save_path)
