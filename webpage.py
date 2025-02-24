from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import pickle
from tensorflow.keras import models
import cv2

app = Flask(__name__)

model_1, model_2, model_3, labels = None, None, None, None

def init():
    global model_1, model_2, model_3, labels
    
    if labels is None:
        labels = pickle.load(open('labels.pkl', 'rb'))

    # if model_1 is None:
    #     model_1 = models.load_model('best_model_1.keras')
        
    if model_2 is None:
        model_2 = models.load_model('best_model_2.keras')
        
    if model_3 is None:
        model_3 = models.load_model('best_model_3.keras')

@app.route('/', methods=['GET', 'POST'])
def index():
    init()

    if 'n3' not in request.files:
        return render_template('index.html', result="Not found")

    try:
        file = request.files['n3']

        if file.filename == '':
            return render_template('index.html', result="No selected file")

        # Load and preprocess the image
        image_array = np.array(Image.open(file.stream)) / 255.0
        image_array = cv2.resize(image_array, (224, 224))

        # Ensure the image has the correct shape (1, 224, 224, 3)
        if image_array.ndim == 2:  # Grayscale image
            image_array = np.stack((image_array,) * 3, axis=-1)  # Convert to RGB
        elif image_array.shape[2] == 4:  # RGBA image
            image_array = image_array[:, :, :3]  # Drop the alpha channel

        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict 1
        # pred_1 = model_1.predict(image_array)
        pred_2 = model_2.predict(image_array)
        pred_3 = model_3.predict(image_array)

        # voting = np.mean([pred_1, pred_2, pred_3], axis=1)
        voting = np.mean([pred_2, pred_3], axis=1)

        label = labels[np.argmax(voting[0])]
        
        msg = f'Prediction: {label}'
        msg +='\n'
        msg += f'{np.max(voting[0]) * 100:.2f}%'
        
        return render_template('index.html', result=msg)

    except Exception as e:
        msg = f'Error has occurred: {e}'
        print(msg)
    return render_template('index.html', result = f"Error:\n{e}")

    
if __name__ == '__main__':
    app.run(debug=True)
    
    
    
