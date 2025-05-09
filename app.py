from flask import Flask, request, jsonify, render_template
import numpy as np
from keras.models import load_model
import os

app = Flask(__name__)
decoder = load_model('decoder.keras')

def scale(wf):
    wf = np.array(wf)
    wf_min, wf_max = np.min(wf), np.max(wf)
    return np.zeros_like(wf) if wf_max == wf_min else 2 * (wf - wf_min) / (wf_max - wf_min) - 1

def moving_average(x, w):
    padded_x = np.pad(x, (w // 2, w - 1 - w // 2), mode='edge')
    return np.convolve(padded_x, np.ones(w), 'valid') / w

def decode(x_ls, y_ls):
    latent = np.array([[x_ls, y_ls]])
    wf = np.squeeze(decoder.predict(latent,verbose=0))
    wf = scale(moving_average(wf, 3))
    return wf

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    x, y = float(data['x']), float(data['y'])
    amp = float(data.get('amp', 1.0))
    freq = float(data.get('freq', 1.0))
    offset = float(data.get('offset', 0.0))

    wf = decode(x, y)
    n_samples = len(wf)
    t = np.linspace(0, 1/freq, n_samples)


    # Apply transformations
    wf_single = amp * wf + offset
    wf_plot = np.tile(wf_single, 3)  # Repeat for 3 periods

    return jsonify(wf=wf_plot.tolist(), wf_single=wf_single.tolist())

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # use 5000 locally, or Render's port in deployment
    app.run(host='0.0.0.0', port=port)