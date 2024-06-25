import os
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import backend as K
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def setup_directories(root_dir):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    alpha = 'a'
    for i in range(26):
        letter_dir = os.path.join(root_dir, alpha)
        if not os.path.exists(letter_dir):
            os.mkdir(letter_dir)
        alpha = chr(ord(alpha) + 1)

def copy_files_to_subdirectories(src_dir, dest_dir):
    for file in os.listdir(src_dir):
        file_path = os.path.join(src_dir, file)
        if os.path.isfile(file_path):
            letter = file[0]
            copyfile(file_path, os.path.join(dest_dir, letter, file))

def create_generators(root_dir, target_size=(28, 28)):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(root_dir,
                                                  target_size=target_size,
                                                  subset='training')

    val_generator = datagen.flow_from_directory(root_dir,
                                                target_size=target_size,
                                                subset='validation')
    return train_generator, val_generator

def build_model(input_shape=(28, 28, 3)):
    K.clear_session()

    entry = L.Input(shape=input_shape)
    x = L.SeparableConv2D(64, (3, 3), activation='relu')(entry)
    x = L.MaxPooling2D((2, 2))(x)
    x = L.SeparableConv2D(128, (3, 3), activation='relu')(x)
    x = L.MaxPooling2D((2, 2))(x)
    x = L.SeparableConv2D(256, (2, 2), activation='relu')(x)
    x = L.GlobalMaxPooling2D()(x)
    x = L.Dense(256)(x)
    x = L.LeakyReLU()(x)
    x = L.Dense(64, kernel_regularizer=l2(2e-4))(x)
    x = L.LeakyReLU()(x)
    x = L.Dense(26, activation='softmax')(x)

    model = Model(entry, x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, train_generator, val_generator, epochs=500):
    model_ckpt = ModelCheckpoint('BrailleNet.keras', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(patience=8, verbose=0)
    early_stop = EarlyStopping(patience=15, verbose=1)

    history = model.fit(train_generator,
                        validation_data=val_generator,
                        epochs=epochs,
                        callbacks=[model_ckpt, reduce_lr, early_stop],
                        verbose=1)
    return history

def plot_history(history):
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()
    plt.savefig('LossVal_loss')

    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()
    plt.savefig('AccVal_acc')

def predict_and_display(model, img_path):
    img = load_img(img_path, target_size=(28, 28))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    classes = 'abcdefghijklmnopqrstuvwxyz'

    predicted_character = classes[predicted_class[0]]

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(f'Predicted Character: {predicted_character}')
    plt.axis('off')
    plt.show()

    with open('prediction_output.txt', 'w') as f:
        f.write(f'Predicted Character: {predicted_character}\n')

    return predicted_character

def main():
    root_dir = './images/'
    src_dir = 'C:/Semester_6/Pengelolaan_Citra_Digital/uas_project/images'
    setup_directories(root_dir)
    copy_files_to_subdirectories(src_dir, root_dir)

    train_generator, val_generator = create_generators(root_dir)

    model = build_model()
    history = train_model(model, train_generator, val_generator)

    model = load_model('BrailleNet.keras')
    plot_history(history)

    img_path = 'C:/Semester_6/Pengelolaan_Citra_Digital/uas_project/test/huruf braille w.png'  # Replace with the path to your image
    predict_and_display(model, img_path)

    model.summary()
    return model

if __name__ == "__main__":
    main()