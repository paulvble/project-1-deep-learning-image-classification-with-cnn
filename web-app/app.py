import os
from flask import Flask, request, redirect, url_for, render_template
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
loaded_model = tf.keras.models.load_model('cifar10_vgg16_model.h5')
#loaded_model = tf.keras.models.load_model('/path/to/cifar10_vgg16_model.h5')


# Define CIFAR-10 class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './static/uploads/'

# Function to process the uploaded image
def process_image(image_path):
    img = Image.open(image_path)
    img = img.resize((32, 32))  # Resize image to match input shape
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/', methods=['GET', 'POST'])

def upload_file():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty file without filename
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Create the uploads directory if it doesn't exist
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            
            # Save the uploaded file to the uploads directory
            filename = file.filename
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            
            # Process the uploaded image
            img_array = process_image(file_path)
            
            # Make prediction using the loaded model
            predictions = loaded_model.predict(img_array)
            predicted_class = np.argmax(predictions)
            class_name = class_names[predicted_class]
            
            return render_template('result.html', filename=filename, class_name=class_name)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=3000)