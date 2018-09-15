import json
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.xception import xception
from keras.utils import to_categorical
from keras.layers import Dense, Flatten
from keras.models import Model


path = "E:/AIchallenge/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json"
class_id = "disease_class"
img_id = "image_id"


def extract_info(path, class_id, img_id):
    with open(path) as load_f:
        id_dict = json.load(load_f)

    train_label = []
    train_img_id = []
    for data in id_dict:
        train_img_id.append(data[img_id])
        train_label.append(data[class_id])

    return train_label, train_img_id


def load_dataset(img_num=100):

    root_path = "E:/AIchallenge/AgriculturalDisease_trainingset/images/"
    train_label, train_img_id = extract_info(path, class_id, img_id)
    train = []
    for img in train_img_id[:img_num]:
        train.append(img_to_array(load_img(root_path + img).resize((300, 300))))

    train = np.array(train)
    train_label = train_label[:img_num]
    return train, train_label


if __name__ == "__main__":
    train, train_label = load_dataset()
    train = xception.preprocess_input(train)
    train_label = to_categorical(train_label)
    base_model = xception.Xception(include_top=False, input_shape=(300, 300, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    print(train.shape)
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], )
    model.fit(train, train_label, epochs=1, batch_size=5)









