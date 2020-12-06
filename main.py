import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,BatchNormalization,Dropout,Activation
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
if __name__ == '__main__':
    train_data_dir = r"C:\Users\42070\Documents\Tencent Files\420709076\FileRecv\2020CV表情识别实验\train"

    val_data_dir = r"C:\Users\42070\Documents\Tencent Files\420709076\FileRecv\2020CV表情识别实验\validation"

    num_features = 64
    num_labels = 7
    batch_size = 64
    epochs = 100
    width, height = 48, 48
    train_datagen = ImageDataGenerator(

           rescale=1./255,

           shear_range=0.2,

           horizontal_flip=True)

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(width, height),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(width, height),
        color_mode="grayscale",
        batch_size=batch_size,
    )

#     # desinging the CNN
#     model = Sequential()

#     model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1),
#                      data_format='channels_last', kernel_regularizer=l2(0.01)))
#     model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#     model.add(Dropout(0.5))

#     model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#     model.add(Dropout(0.5))

#     model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#     model.add(Dropout(0.5))

#     model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#     model.add(Dropout(0.5))

#     model.add(Flatten())

#     model.add(Dense(2 * 2 * 2 * num_features, activation='relu'))
#     model.add(Dropout(0.4))
#     model.add(Dense(2 * 2 * num_features, activation='relu'))
#     model.add(Dropout(0.4))
#     model.add(Dense(2 * num_features, activation='relu'))
#     model.add(Dropout(0.5))

#     model.add(Dense(num_labels, activation='softmax'))
    model = Sequential()
    
    model.add(Conv2D(64, (5, 5), padding='same',
                     input_shape=(width, height, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    model.summary()

    # Compliling the model with adam optimixer and categorical crossentropy loss
    model.compile(loss=categorical_crossentropy,
                  optimizer='Adam',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=int(17591/64/7),
        epochs=epochs,
        verbose=1,
        validation_data=val_generator,
        validation_steps=int(7533/64/7),
        shuffle=True
    )
    # print(val_generator.classes)
    # print(val_generator.image_shape)
    # saving the  model to be used later
    fer_json = model.to_json()
    with open("fer.json", "w") as json_file:
        json_file.write(fer_json)
    model.save_weights("fer.h5")
    print("Saved model to disk")