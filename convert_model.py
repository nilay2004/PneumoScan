import os
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

def convert_model():
    try:
        # Define custom objects to handle the old model format
        custom_objects = {
            'InputLayer': InputLayer,
            'Dense': Dense,
            'Conv2D': Conv2D,
            'MaxPooling2D': MaxPooling2D,
            'Flatten': Flatten,
            'Dropout': Dropout,
            'BatchNormalization': BatchNormalization
        }
        
        # Load the model with custom objects
        input_path = os.path.join('models', 'vgg16_model.keras')
        output_path = os.path.join('models', 'vgg16_model_new.keras')
        
        print(f"Loading model from {input_path}")
        model = tf.keras.models.load_model(input_path, custom_objects=custom_objects, compile=False)
        
        # Create a new model with the same architecture
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = model(inputs)
        new_model = tf.keras.Model(inputs=inputs, outputs=x)
        
        # Compile the new model
        new_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Saving model to {output_path}")
        new_model.save(output_path)
        print("Model converted and saved successfully!")
        
    except Exception as e:
        print(f"Error converting model: {str(e)}")
        raise

if __name__ == "__main__":
    convert_model() 