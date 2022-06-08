import imp
import cv2
import time
from keras.preprocessing.image import img_to_array
import numpy as np
import os
from tensorflow.keras import layers
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from keras.models import Model

import tensorflow as tf
# from keras.utils.vis_utils import plot_model
import sys
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask

class acRec:
    def __init__(self,modelName):
        self.h=224
        self.w=224
        self.shp=(self.w,self.h)
        self.modeName=modelName
    def arr(self,array):
        array=np.array(array)
        return array
    def vedioConvert(self,path):
        cap = cv2.VideoCapture(path)
        tic = time.perf_counter()
        i=0
        frames = np.zeros((50, self.h, self.w, 3), dtype=np.float)
        while cap.isOpened():
            _, frame = cap.read()
            try:
                RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                break
            img = cv2.resize(RGB,self.shp)
            img = np.expand_dims(img,axis=0)
            frames[i][:] = img
            if cv2.waitKey(1) == ord('q'):
                break
            i+=1
        toc = time.perf_counter()
        cap.release()
        cv2.destroyAllWindows()
        return [frames,(((path.split('/'))[-1]).split('_'))[0]]
    def drictor(self,path):
        dataTrain=np.zeros((126,50, self.h, self.w, 3), dtype=np.float)
        labelTrain=[]
        
        for dr in os.listdir(path):
            i=0
            for video in os.listdir(f'{path}/{dr}'):
                data=self.vedioConvert(f'{path}/{dr}/{video}')
                dataTrain[i][:]=data[0]
                labelTrain.append(data[1])
                i+=1
                    
        return self.arr(dataTrain),self.arr(labelTrain)
    
    def build_convnet(shape=(224, 224, 3)):
        momentum = .9
        model = keras.Sequential()
        model.add(layers.Conv2D(32, (3,3), input_shape=shape,
            padding='same', activation='relu'))
        model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization(momentum=momentum))

        model.add(layers.MaxPool2D())

        model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
        model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization(momentum=momentum))

        model.add(layers.MaxPool2D())

        model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
        model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization(momentum=momentum))

        # flatten...
        model.add(layers.GlobalMaxPool2D())
        return model
    def action_model(shape,classes):

        convnet = build_convnet(shape[1:])
        model = keras.Sequential()    
        model.add(layers.TimeDistributed(convnet, input_shape=shape))    
        model.add(layers.LSTM(30))    
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(.5))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(.5))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(.5))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(classes, activation='softmax'))
        return model
    def define_model(self):
        classes=5

        shape=(sequence_length,self.h, self.w,3)
        modle=action_model(shape,classes)
        model.compile(
            optimizer="adam",
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        
    def define_model2(self):
        IMG_SIZE = 224
        embed_dim = 2048
#         sequence_length = 50
        dense_dim = 4
        num_heads = 1
        
        inputs = keras.Input(shape=(self.h, self.w,3))
        x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(inputs)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(units=512, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.classes, activation="softmax")(x)
        model = Model(inputs, outputs)

        model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        return model
    def define_model3(self):
        IMG_SIZE = 224
        embed_dim = 2048
        sequence_length = 50
        dense_dim = 4
        num_heads = 1
         
        inputs = keras.Input(shape=(self.h, self.w,3))
        x = PositionalEmbedding(
            sequence_length, embed_dim, name="frame_position_embedding"
        )(inputs)
        x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(0.5)(x)
        
        outputs = layers.Dense(self.classes, activation="softmax")(x)
        model = Model(inputs, outputs)

        model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        return model
    
    
    def sict(self):
        inputs = keras.Input(shape=(self.h, self.w,3))
        
        model=layers.Conv2D(32,3,padding="same", activation="relu" )(inputs)
        model=layers.Conv2D(32,3,padding="same", activation="relu" )(inputs)
        
        model=layers.Conv2D(32,3,padding="same", activation="relu" )(inputs)
        model=layers.Conv2D(32,3,padding="same", activation="relu" )(inputs)
        model=layers.MaxPool2D()(model)
        
        model=layers.Conv2D(64, 3, padding="same", activation="relu")(model)        
        model=layers.Conv2D(64, 3, padding="same", activation="relu")(model)
        
        model=layers.Conv2D(64, 3, padding="same", activation="relu")(model)        
        model=layers.Conv2D(64, 3, padding="same", activation="relu")(model)
        model=layers.MaxPool2D()(model)
        model=layers.Dropout(0.5)(model)

        model=layers.Flatten()(model)
        model=layers.Dense(128,activation="relu")(model)
        model=layers.Dense(self.classes, activation="softmax")(model)
        model = Model(inputs, model)
        
        opt = SGD(learning_rate=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        
        opt = SGD(learning_rate=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    
    def vgg16(self):
        model = VGG16(include_top=False, input_shape=(224, 224, 3))
        # mark loaded layers as not trainable
        for layer in model.layers:
            layer.trainable = False
        # add new classifier layers
        flat1 = layers.Flatten()(model.layers[-1].output)
        class1 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
        output = layers.Dense(self.classes, activation='sigmoid')(class1)
        # define new model
        model = Model(inputs=model.inputs, outputs=output)
        # compile model
        opt = SGD(learning_rate=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def simple(self):
        inputs = keras.Input(shape=(self.h, self.w,3))
        x = layers.Conv2D(filters=64, kernel_size=2, activation="relu")(inputs)
        x = layers.Conv2D(filters=64, kernel_size=2, activation="relu")(x)
        x = layers.MaxPool2D(pool_size=3)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(filters=128, kernel_size=2, activation="relu")(x)
        x = layers.Conv2D(filters=128, kernel_size=2, activation="relu")(x)
        x = layers.MaxPool2D(pool_size=3)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(filters=256, kernel_size=2, activation="relu")(x)
        x = layers.Conv2D(filters=256, kernel_size=2, activation="relu")(x)
        x = layers.MaxPool2D(pool_size=3)(x)
        x = layers.BatchNormalization()(x)
        
#         x = layers.Dense(units=3, activation="relu")(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(units=128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(self.classes, activation="softmax")(x)
        
        model = Model(inputs, x)
        
        opt = SGD(learning_rate=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        
        return  model
    
    def modelFit(self,model,train_it,test_it):
#         checkpoint = keras.callbacks.ModelCheckpoint(
#             'poseBasedAR_V1', save_weights_only=True, save_best_only=True, verbose=1
#             )
        
#         self.history = model.fit( [train_data], train_labels, validation_split=0.15, epochs=3, callbacks=[checkpoint],)
        self.history = model.fit(train_it, steps_per_epoch=len(train_it),validation_data=test_it, validation_steps=len(test_it), epochs=3, verbose=1)
        _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
        print(f"Test accuracy: {round(acc * 100, 2)}%")
        model.save(str(self.modeName)+'.h5')
        return model
        
        
    def summarize_diagnostics(history):
        plt.subplot(211)
        plt.title('Кросс энтропийн алдагдал Cross Entropy Loss')
        plt.plot(history.history['loss'], color='blue', label='train')
        plt.plot(history.history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(212)
        plt.title('Ангиллын нарийвчлал')
        plt.plot(history.history['accuracy'], color='blue', label='train')
        plt.plot(history.history['val_accuracy'], color='orange', label='test')
        # save plot to file
        filename = sys.argv[0].split('/')[-1]
        plt.savefig(filename + '_plot.png')
        plt.show()
        plt.close()

    def processVGG16(self,path):
#         dataTrain,labelTrain=self.drictor(path)
        datagen = ImageDataGenerator()
        train_it=datagen.flow_from_directory(path+'/train',class_mode='categorical', batch_size=64, target_size=(224, 224))
        className=[]
        for i in train_it.class_indices:
            className.append(i)
        self.classNameIndex=className
        test_it=datagen.flow_from_directory(path+'/test',class_mode='categorical', batch_size=8, target_size=(224, 224))
        print(train_it.num_classes)
        self.classes=train_it.num_classes
        
#         model=self.sict()

#         model=self.simple()
        model=self.vgg16()
        model.summary() 
        # plot_model(model, to_file='model_plot.png', show_shapes=False, show_layer_names=True)
        model=self.modelFit(model,train_it,test_it)
        # self.summarize_diagnostics(self.history)
        return model
    def processsimple(self,path):
#         dataTrain,labelTrain=self.drictor(path)
        datagen = ImageDataGenerator()
        train_it=datagen.flow_from_directory(path+'/train',class_mode='categorical', batch_size=64, target_size=(224, 224))
        className=[]
        for i in train_it.class_indices:
            className.append(i)
        self.classNameIndex=className
        test_it=datagen.flow_from_directory(path+'/test',class_mode='categorical', batch_size=8, target_size=(224, 224))
        print(train_it.num_classes)
        self.classes=train_it.num_classes
        
#         model=self.sict()

        model=self.simple()
        model.summary() 
        # plot_model(model, to_file='model_plot.png', show_shapes=False, show_layer_names=True)
        model=self.modelFit(model,train_it,test_it)
        # self.summarize_diagnostics(self.history)
        return model
    def processsict(self,path):
#         dataTrain,labelTrain=self.drictor(path)
        datagen = ImageDataGenerator()
        train_it=datagen.flow_from_directory(path+'/train',class_mode='categorical', batch_size=64, target_size=(224, 224))
        className=[]
        for i in train_it.class_indices:
            className.append(i)
        self.classNameIndex=className
        test_it=datagen.flow_from_directory(path+'/test',class_mode='categorical', batch_size=8, target_size=(224, 224))
        print(train_it.num_classes)
        self.classes=train_it.num_classes
        
        model=self.sict()
        model.summary() 
        # plot_model(model, to_file='model_plot.png', show_shapes=False, show_layer_names=True)
        model=self.modelFit(model,train_it,test_it)
        # self.summarize_diagnostics(self.history)
        return model
    