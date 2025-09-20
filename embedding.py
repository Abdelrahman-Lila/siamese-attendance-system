import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K


def preprocess_image(image_path):
    try:
        # Read and decode image
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        # Resize to 224x224 (ResNet50 input)
        img = tf.image.resize(img, [224, 224])
        # Convert to float32 and normalize to [0, 1]
        img = tf.cast(img, tf.float32) / 255.0
        # Apply ResNet50 preprocessing
        img = tf.keras.applications.resnet50.preprocess_input(img * 255.0)
        return img
    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {str(e)}")

# Define function to get embedding
def get_embedding(image_path, model):
    try:
        img = preprocess_image(image_path)
        img = tf.expand_dims(img, axis=0)
        # Get embedding (512D)
        embedding = model(img, training=False)
        return embedding.numpy()
    except Exception as e:
        raise ValueError(f"Error getting embedding for {image_path}: {str(e)}")


def cosine_similarity(vectorA, vectorB):
    vectorA = tf.nn.l2_normalize(vectorA, axis=1)
    vectorB = tf.nn.l2_normalize(vectorB, axis=1)
    return tf.reduce_sum(vectorA * vectorB, axis=1, keepdims=True)


# Define function to compare two images
def compare_images(image_path1, image_path2, model, threshold=0.6):
    try:
        # Get embeddings
        embed1 = get_embedding(image_path1, model)
        embed2 = get_embedding(image_path2, model)

        cosine_similarity = cosine_similarity(embed1, embed2)

        # Compute Euclidean distance
        distance = np.linalg.norm(embed1 - embed2)

        # Predict same/different
        prediction = "Same person" if distance < threshold else "Different person"

        return distance, prediction, cosine_similarity
    except Exception as e:
        raise ValueError(f"Error comparing images: {str(e)}")



print(os.path.getsize("/content/siamese_try.keras"))

model_path = "/content/siamese_try.keras"
try:
    backbone = tf.keras.models.load_model(model_path, safe_mode=False)
    print(f"Loaded model from {model_path}")
except Exception as e:
    raise ValueError(f"Error loading model from {model_path}: {str(e)}")

image_path1 = "/content/face_1.jpg"
image_path2 = "/content/face_2.jpg"
image_path3 = "/content/face_3.jpg"

if not (os.path.exists(image_path1) and os.path.exists(image_path2)):
    raise ValueError(
        f"One or both image paths do not exist: {image_path1}, {image_path2}"
    )

distance, prediction, cosine = compare_images(image_path1, image_path2, backbone)
print(f"Euclidean Distance: {distance:.4f}")
print(f"Prediction: {prediction}")
print(f"cosine similarity: {cosine}")
