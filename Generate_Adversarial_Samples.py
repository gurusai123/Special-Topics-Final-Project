import tensorflow as tf

# Enable GPU globally
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to prevent TensorFlow from allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Optionally, make all GPUs visible
        tf.config.set_visible_devices(gpus, 'GPU')

        print(f"Using {len(gpus)} GPU(s): {gpus}")
    except RuntimeError as e:
        print(f"Error during GPU setup: {e}")
else:
    print("No GPU detected, running on CPU.")

from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import csv
from PIL import Image
import os
import pandas as pd

# !pip install foolbox
import foolbox

mpl.rcParams['figure.figsize'] = (5, 5)
mpl.rcParams['axes.grid'] = False


# Defining a function to Preprocess image
def preprocessImage(image):
    image = image.resize((32, 32)) # resize images
    return image


# Defining a function to read images
def readTestTrafficSigns(cnt1):
    y_test = pd.read_csv(r'C:\Users\gurus\Downloads\archive (1)\Test.csv')
    
    labels = y_test["ClassId"].values
    imgs = y_test["Path"].values
    
    data=[]
    
    cnt = 0
    
    with tf.device('/GPU:0'):
        for img in imgs:
            cnt+=1
            raw_image = Image.open(r'C:\Users\gurus\Downloads\archive (1)\\'+img)
            image = raw_image.resize([32, 32])
            image_array = np.array(image)
            data.append(np.array(image_array))
            if(cnt == cnt1):
                break
    X_test=np.array(data)
    return X_test, labels
    

def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = image/255
    image = tf.image.resize(image, (32, 32))
    image = image[None, ...]
    return image

# Loading images
pretrained_model = tf.keras.models.load_model(r"C:\Users\gurus\Downloads\models\vggnet.h5")
pretrained_model.trainable = False
images, labels = readTestTrafficSigns(20)

# Converting labels to One Hot Encoded vectors
labels = to_categorical(labels, num_classes=43)

np.shape(images),np.shape(labels)

# Select an image to create adversarial samples to visualize
image_index = 3
image_array = images[image_index]
label = labels[image_index]
image = images[image_index]
image = preprocess(image)

predicted_label = K.eval(pretrained_model(image))
lbl = np.argmax(predicted_label[0])
real_lbl = np.argmax(label) 
plt.title("Speed Limit 30\nreal class: %d predicted class: %d with %f confidence" % (real_lbl, lbl, predicted_label.max()))
plt.imshow(image_array.reshape((32,32,3)))
plt.show()


loss_object = tf.keras.losses.CategoricalCrossentropy()


# Defining a function to create adversarial image samples
def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad

label = np.expand_dims(label, axis=0)

perturbations = create_adversarial_pattern(image, label)

array = perturbations.numpy()
plt.imshow(array.reshape((32, 32, 3)))
plt.show()

def display_images(image, description):
    label = pretrained_model(image)
    lbl = np.argmax(label[0])

    plt.figure()
    plt.title(
        "%s\nPredicted class: %d with %f confidence" % (description, lbl, label[0][lbl])
    )
    plt.imshow(image[0].reshape((32, 32, 3)))
    plt.show()
    
epsilons = [0, 0.01, 0.1, 0.15]
descriptions = [f"Epsilon = {eps:.3f}" if eps else "Input" for eps in epsilons]

for i, eps in enumerate(epsilons):
    adv_x = image + eps * perturbations
    adv_x = tf.clip_by_value(adv_x, 0, 1)
    adv_x = adv_x.numpy()
    display_images(adv_x, descriptions[i])

# Saving adversarial images
def save_images(image, image_no, dir_name, real_label, real_predicted, real_predicted_conf):
    
    
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    # Getting the label using the pretrained model
    label = pretrained_model.predict(image)
    label = K.eval(label)
    lbl = np.argmax(label[0])

    # Creating a matplotlib figure
    fig, ax = plt.subplots()
    ax.set_axis_off()
    fig.add_axes(ax)

    if isinstance(image, tf.Tensor):
        reshaped_image = image.numpy()
    else:
        reshaped_image = image

    # print("Raw reshaped_image shape:", reshaped_image.shape)

    # Handling different image shapes
    if reshaped_image.ndim == 4 and reshaped_image.shape[0] == 1:
        reshaped_image = reshaped_image[0]
    if reshaped_image.ndim == 3 and reshaped_image.shape[-1] == 3:
        pass
    elif reshaped_image.ndim == 2:
        reshaped_image = np.stack([reshaped_image] * 3, axis=-1)
    else:
        raise ValueError(f"Unexpected image shape: {reshaped_image.shape}")

    # print("Reshaped image shape:", reshaped_image.shape)

    # Normalizing the pixel values to [0, 1]
    reshaped_image = (reshaped_image - np.min(reshaped_image)) / (
        np.max(reshaped_image) - np.min(reshaped_image) + 1e-10
    )
   

    # Displaying and saving the images by setting up the dpi
    ax.imshow(reshaped_image, cmap='gray' if reshaped_image.ndim == 2 else None)
    dpi = 100
    image_path = os.path.join(dir_name, f"{image_no:04d}.png")
    fig.savefig(image_path, dpi=dpi)
    plt.close(fig)

    # Create or append to the CSV file
    csv_file = os.path.join(dir_name, 'info.csv')
    row = [
        f"{image_no:04d}.png",
        real_label,
        real_predicted,
        real_predicted_conf,
        lbl,
        label[0].max()
    ]

    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['File Name', 'Real Label', 'Real Predicted', 'Confidence', 
                             'Adversarial Predicted Label', 'Adversarial Confidence'])
        writer.writerow(row)

    print(f"Saved image {image_no:04d}.png and updated {csv_file}")

images2, labels2 = readTestTrafficSigns(1500)

labels2 = to_categorical(labels2, num_classes=43)

epsilons = [0.01,0.10,0.15]
dir_names = [('epsilon_{:d}'.format(int(eps*100))) for eps in epsilons]

def preprocess_image(image):
    image = image / 255.0
    if len(image.shape) == 2 or image.shape[-1] == 1:
        image = np.stack((image, image, image), axis=-1)
    return image


for image_ind in range(len(images2)):
    image_array2 = images2[image_ind]
    label2 = labels2[image_ind]

    image2 = preprocess_image(image_array2)
    image2 = np.expand_dims(image2, axis=0)
    label2 = np.expand_dims(label2, axis=0)
    
    predicted_label2 = pretrained_model.predict(image2)
    lbl2 = np.argmax(predicted_label2[0])
    real_lbl2 = np.argmax(label2[0])

    image2 = tf.convert_to_tensor(image2)
    
    perturbations2 = create_adversarial_pattern(image2, label2)

    for i, eps in enumerate(epsilons):
        adv_x = image2 + eps * perturbations2
        adv_x = tf.clip_by_value(adv_x, 0, 1)
        
        adv_x_np = adv_x.numpy()
        # print(np.shape(adv_x_np))
        reshaped_image = adv_x_np[0]

        reshaped_image = (reshaped_image - reshaped_image.min()) / (reshaped_image.max() - reshaped_image.min() + 1e-10)
        reshaped_image = tf.convert_to_tensor(reshaped_image)
        # print("Saving image with shape:", reshaped_image.shape)

        save_images(
            reshaped_image,
            image_ind,
            dir_names[i],
            real_lbl2,
            lbl2,
            predicted_label2[0].max()
        )

    print("Image %d is done" % image_ind)