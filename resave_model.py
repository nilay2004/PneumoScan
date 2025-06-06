import os
import tensorflow as tf
import logging

def resave_model():
    try:
        # Use os.path.join for proper path handling
        input_path = os.path.join('models', 'vgg16_model.keras')
        output_path = os.path.join('models', 'vgg16_model_new.keras')
        
        # Load the model
        print(f"Loading model from {input_path}")
        model = tf.keras.models.load_model(input_path)
        
        # Save the model
        print(f"Saving model to {output_path}")
        model.save(output_path)
        print("Model re-saved successfully!")
        
    except Exception as e:
        print(f"Error re-saving model: {str(e)}")
        raise

if __name__ == "__main__":
    resave_model() 