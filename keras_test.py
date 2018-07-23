import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import (Input, Conv1D, BatchNormalization, GlobalMaxPooling1D, 
                          Activation, Permute, Conv2D, GlobalMaxPooling2D)
import dataset
reload(dataset)

n_classes = 10


iterator = dataset.dataset_input_fn(True, 64)
sounds, lbls = iterator.get_next()
lbls = tf.one_hot(lbls, n_classes)


inputs = Input(tensor=sounds)


def my_cnn(x):
    for i in range(5):
        print(x)
        x = Conv1D(32, 3, strides=2, padding='valid', activation='relu')(x)
        x = BatchNormalization()(x)
    
    x = keras.layers.Reshape((-1, 32, 1))(x)
    for i in range(4):
        print(x)
        x = Conv2D(32, (3, 3))(x)
        x = BatchNormalization()(x)

    x = Conv2D(10, (3, 3))(x)
    x = GlobalMaxPooling2D()(x)
    y = Activation('softmax')(x)

    return y

model = Model(inputs=inputs, outputs=my_cnn(inputs))

optimizer = keras.optimizers.SGD(lr=0.003)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              target_tensors=[lbls])

model.summary()

keras.backend.get_session().run(iterator.initializer)
model.fit(epochs=5, steps_per_epoch=50)  # starts training


weight_path = os.path.join('./saved_wt.h5')
model.save_weights(weight_path)
s, l = keras.backend.get_session().run([sounds, lbls])

# Clean up the TF session.
K.clear_session()

# Second session to test loading trained model without tensors.
x_test = s.astype(np.float32)
y_test = l[:, 0, :]

x_test_inp = Input(shape=x_test.shape[1:])
test_out = my_cnn(x_test_inp)
test_model = keras.models.Model(inputs=x_test_inp, outputs=test_out)

test_model.load_weights(weight_path)
test_model.compile(optimizer=optimizer,
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
test_model.summary()

loss, acc = test_model.evaluate(x_test, y_test, n_classes)

test_model.predict(x_test)
