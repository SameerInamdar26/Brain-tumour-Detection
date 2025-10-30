import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
import matplotlib.pyplot as plt

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate a Grad-CAM heatmap for a given image and model prediction.
    
    Args:
        img_array (numpy.ndarray): Preprocessed image array of shape (1, H, W, C).
        model (tf.keras.Model): Trained Keras model.
        last_conv_layer_name (str): Name of the last convolutional layer.
        pred_index (int, optional): Index of the predicted class. Defaults to None.
    
    Returns:
        numpy.ndarray: Heatmap array normalized between 0 and 1.
    """
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, size, alpha=0.4):
    """
    Superimpose the Grad-CAM heatmap on the original image.
    
    Args:
        img_path (str): Path to the original image.
        heatmap (numpy.ndarray): Grad-CAM heatmap.
        size (tuple): Target size (W, H) for resizing.
        alpha (float): Intensity factor for overlay.
    
    Returns:
        PIL.Image.Image: Image with Grad-CAM overlay.
    """
    # Load original image
    img = keras_image.load_img(img_path, target_size=size)
    img = keras_image.img_to_array(img)

    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = keras_image.array_to_img(jet_colors[heatmap])
    jet_heatmap = jet_heatmap.resize(size)
    jet_heatmap = keras_image.img_to_array(jet_heatmap)

    # Superimpose heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras_image.array_to_img(superimposed_img / 255.0)

    return superimposed_img
 
