import os
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay

from tensorflow.keras.preprocessing.image import array_to_img, img_to_array,load_img
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Activation, Flatten, Dropout, BatchNormalization
import keras_tuner

from tqdm import tqdm

from itertools import cycle



# from tensorflow.keras.applications import densenet 


# Visualization of site distributions

def site_distributions(train, val, normalize = False):   
    fig, ax = plt.subplots(1,2, figsize = (12,5), sharey=True)
    
    if normalize == False:
        y = "count"
    else:
        y = "proportion"
    
    g0 = sns.barplot(data = train["site"].value_counts(normalize=normalize).reset_index(), 
                        x = "site", 
                        y = y, 
                        color = "crimson", ax = ax[0])
    
    g0.set_xticks(list(range(0, len(train["site"].value_counts()), 20)))
    g0.set_xlabel("Training sites")
    

    g1 = sns.barplot(data = val["site"].value_counts(normalize=normalize).reset_index(), 
                    x = "site", 
                    y = y, 
                    color = "blue", ax = ax[1])

    g1.set_xticks(list(range(0, len(val["site"].value_counts()), 5)))
    g1.set_xlabel("Validation sites")

    plt.tight_layout()



# Visualization training loss and accuracy

def plot_metric(history):
    plt.plot(history.history["accuracy"], label = "accuracy")
    plt.plot(history.history["val_accuracy"], label = "val_accuracy")
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss (Cross Entropy)')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)



# Creating 4D tensors from images

def path_to_tensor(img_path, size_x = 224, size_y = 224):
    # loads RGB image as PIL.Image.Image type
    img = load_img(img_path, target_size=(size_x, size_y))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = img_to_array(img)
    
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)
    
    

def paths_to_tensor(img_paths, size_x = 224, size_y = 224):
    list_of_tensors = [path_to_tensor(img_path, size_x, size_y) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)





# stratification by species and site exclusivity

def stratified_train_val_split(train_features:pd.DataFrame, train_labels:pd.DataFrame, random_state = 52):
    
    train_total = train_features.merge(train_labels, how="left", on="id")
    
    train_site_group = train_total.groupby("site").sum().loc[:,train_labels.columns]

    # Get the most frequently counted species by site
    train_site_group['stratify_col'] = train_site_group.idxmax(axis=1)

    # Stratifying train and validation split by species
    train_sites, val_sites = train_test_split(train_site_group.index, test_size = 0.2, 
                                              random_state = random_state, 
                                              stratify = train_site_group["stratify_col"])
    
    #Get the groups
    train_groups = [train_total[train_total['site'] == site] for site in train_sites]
    val_groups = [train_total[train_total['site'] == site] for site in val_sites]

    # Concatenate the groups to form the final dataframes
    train_df = pd.concat(train_groups)
    val_df = pd.concat(val_groups)

    X_train = train_df.iloc[:,0:2].sort_index()
    X_val = val_df.iloc[:,0:2].sort_index()
    y_train = train_df.iloc[:,2:].sort_index()
    y_val = val_df.iloc[:,2:].sort_index()

    return X_train, X_val, y_train, y_val




# Create sub-directories for each species and copy images from the train_features folder to 
# the sub-directory of each species


def create_species_directory(df_train, df_val, home_path):

    # create train and validation directories
    train_path = os.path.join(home_path,'train')
    os.mkdir(train_path)
    val_path = os.path.join(home_path,'valid')
    os.mkdir(val_path)

    # create sub-directories for each species
    for animal in df_train.columns:
        animal_train_path = os.path.join(home_path + r"/train", animal)
        os.mkdir(animal_train_path)

    for animal in df_train.columns:
        animal_val_path = os.path.join(home_path + r"/valid", animal)
        os.mkdir(animal_val_path)

    
    # copy images into designated sub-directories for train dataset

    dict_species_train = df_train.idxmax(axis = 1).reset_index(name = "species").groupby('species')['id'].apply(list).to_dict()

    for species, image_ids in dict_species_train.items():
        for image_id in image_ids:

            source_path_train = os.path.join(home_path + r'/train_features/', image_id + '.jpg')
            target_path_train = os.path.join(home_path + r'/train/' + species, image_id + '.jpg')
            
            shutil.copy(src = source_path_train, dst =  target_path_train)
    

    dict_species_val = df_val.idxmax(axis = 1).reset_index(name = "species").groupby('species')['id'].apply(list).to_dict()

    for species, image_ids in dict_species_val.items():
        for image_id in image_ids:

            source_path_val = os.path.join(home_path + r'/train_features/', image_id + '.jpg')
            target_path_val = os.path.join(home_path + r'/valid/' + species, image_id + '.jpg')
            
            shutil.copy(src = source_path_val, dst =  target_path_val)



# Densenet Hypermodel for cross-validation

class DensenetHypermodel(keras_tuner.HyperModel):

  def build (self, hp = keras_tuner.HyperParameters()):


      K.clear_session()

      model = Sequential()

      # Data Augmentation

      model.add(tf.keras.layers.RandomRotation(factor = 0.4, fill_mode = "nearest"))
      model.add(tf.keras.layers.RandomTranslation(height_factor = 0.2, width_factor = 0.2, fill_mode = "nearest"))
      #model.add(keras_cv.layers.RandomShear(x_factor = 0.2, fill_mode = "nearest"))
      model.add(tf.keras.layers.RandomZoom(height_factor = (-0.3, 0.3), fill_mode = "nearest"))
      #model.add(tf.keras.layers.RandomBrightness(factor = 0.1, value_range = [0, 1]))
      #model.add(keras_cv.layers.RandomSaturation(factor = (0.4,0.6)))
      model.add(tf.keras.layers.RandomFlip("horizontal"))


      model.add(densenet_base)

      model.add(Flatten())
      model.add(BatchNormalization())


      #Layer 1

      model.add(Dense(units = hp.Choice("units_1", [64, 128, 256, 512]),
                      kernel_initializer = hp.Choice("kernel_init", ["uniform", "glorot_uniform", "glorot_normal"]),
                      kernel_regularizer = keras.regularizers.L2(hp.Float("kernel_reg_1", min_value=1e-4, max_value=1e-1, sampling = "log")),
                      name = "dense1"))
      if hp.Boolean("dropout_layer_1"):
        model.add(Dropout(rate = hp.Float("dropout_rate_1", min_value = 0, max_value = 0.5, step = 0.1)))
      model.add(BatchNormalization())
      model.add(Activation(hp.Choice("activation", ["relu","gelu","selu"])))


    # Layer 2

      model.add(Dense(units = hp.Choice("units_2", [64, 128, 256, 512]),
                      kernel_initializer = hp.Choice("kernel_init", ["uniform", "glorot_uniform", "glorot_normal"]),
                      kernel_regularizer = keras.regularizers.L2(hp.Float("kernel_reg_2", min_value=1e-4, max_value=1e-1, sampling = "log")),
                      name = "dense2"))
      if hp.Boolean("dropout_layer_2"):
        model.add(Dropout(rate = hp.Float("dropout_rate_2", min_value = 0, max_value = 0.5, step = 0.1)))
      model.add(BatchNormalization())
      model.add(Activation(hp.Choice("activation", ["relu","gelu","selu"])))


      # Layer 3

      #model.add(Dense(units = hp.Choice("units_3", [16, 32, 64, 128, 256]),
      #                kernel_initializer = hp.Choice("kernel_init", ["uniform", "glorot_uniform", "glorot_normal"]),
      #                kernel_regularizer = keras.regularizers.L2(hp.Float("kernel_reg_3", min_value=1e-4, max_value=1e-1, sampling = "log")),
      #                name = "dense3"))
      #if hp.Boolean("dropout_layer_3"):
      #  model.add(Dropout(rate = hp.Float("dropout_rate_3", min_value = 0.1, max_value = 0.5, step = 0.1)))
      #model.add(BatchNormalization())
      #model.add(Activation(hp.Choice("activation", ["relu","gelu","selu"])))



      # Output layer (softmax)
      model.add(Dense(units = 8, activation = "softmax", name = "output"))



      #compiling model

      lr = hp.Float("lr", min_value = 1e-4, max_value = 1e-1, sampling = "log")

      optimizers_dict = {
          "adam":    keras.optimizers.Adam(learning_rate = lr),
          "sgd":     keras.optimizers.SGD(learning_rate = lr, momentum =  hp.Float("momentum", min_value = 0.85, max_value = 0.95)),
          "rmsprop": keras.optimizers.RMSprop(learning_rate = lr)
          }

      hp_optimizers = hp.Choice('optimizer', values=["adam", "sgd", "rmsprop"])


      model.compile(optimizer = optimizers_dict[hp_optimizers], metrics = ["accuracy"],
                    loss = tf.keras.losses.CategoricalCrossentropy())

      return model


  def fit(self, hp, model, *args, **kwargs):
      return model.fit(*args, batch_size = hp.Choice("batch_size", [32, 64, 128, 256]), **kwargs)





# Decode predictions 

def decode_predictions(predictions, label_mapping):

    def get_labels(predictions, label_mapping):
      predicted_indices = np.argmax(predictions, axis=1) 
      predicted_labels = [label_mapping[idx] for idx in predicted_indices]
      return predicted_labels

    predicted_labels = get_labels(predictions, label_mapping)
    
    df = pd.DataFrame({"predicted_labels": predicted_labels})

    df = pd.get_dummies(df, columns=["predicted_labels"], dtype="int32", prefix="", prefix_sep="")
    return df




# custom ROC curve

def roc_curve_total(y_true, y_pred, label_map):
  fig, ax = plt.subplots(figsize=(7, 7))

  colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "yellow", "pink", "green", "brown"])
  for class_id, color in zip(range(8), colors):
      RocCurveDisplay.from_predictions(
          y_true.iloc[:, class_id],
          y_pred.iloc[:, class_id],
          name=label_map[class_id],
          color=color,
          ax=ax,
          plot_chance_level=(class_id == 2),
      )

  _ = ax.set(
      xlabel="False Positive Rate",
      ylabel="True Positive Rate",
      title="Macro-averaged One-vs-Rest\nReceiver Operating Characteristic",
  )