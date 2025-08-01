from flask import Flask, render_template, jsonify
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from datetime import datetime

app = Flask(__name__)

# --- Load Model and Dataset Once ---
print("🔄 Loading model and dataset...")
model = tf.keras.models.load_model('handwriting_rnn_tf.h5', compile=False)
data = np.load("deepwriting_training.npz", allow_pickle=True)
strokes = data['strokes']
mean, std = data['mean'], data['std']
print("✅ Model and dataset loaded.")

# --- Normalization ---
def normalize(seq):
    return (seq - mean) / std

def denormalize(seq):
    return (seq * std) + mean

# --- Sequence Generator ---
def generate_sequence(model, seed_seq, length=150):
    seed_seq = tf.expand_dims(seed_seq, axis=0)  # shape (1, T, 3)
    output = []

    for _ in range(length):
        pred = model(seed_seq, training=False)
        mean_pred = pred[:, -1, :].numpy()[0]

        # ✅ Add randomness + ensure float32 type
        next_pt = np.random.normal(loc=mean_pred, scale=0.1).astype(np.float32)
        output.append(next_pt)

        # Add next point with shape (1, 1, 3)
        next_input = tf.expand_dims(tf.expand_dims(next_pt, axis=0), axis=0)
        seed_seq = tf.concat([seed_seq, next_input], axis=1)

    return np.array(output)

# --- Plot and Save Image ---
def plot_and_save(seq, save_path):
    seq = denormalize(seq)
    x, y = 0, 0
    X, Y = [], []

    fig, ax = plt.subplots(figsize=(6, 2))
    for dx, dy, p in seq:
        x += dx
        y += dy
        if p < 0.5:
            X.append(x)
            Y.append(y)
        else:
            if X:
                ax.plot(X, Y, 'k-', linewidth=1)
            X, Y = [], []
    if X:
        ax.plot(X, Y, 'k-', linewidth=1)

    ax.axis('off')
    ax.invert_yaxis()
    fig.tight_layout()

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate():
    try:
        print("🚀 Generating new handwriting...")

        # Use a random seed every time
        random_index = np.random.randint(0, len(strokes))
        seed = normalize(strokes[random_index][:10].astype(np.float32))

        generated = generate_sequence(model, seed)

        # Create unique filename with timestamp
        timestamp = int(datetime.now().timestamp())
        output_filename = f"output_{timestamp}.png"
        output_path = os.path.join("static", output_filename)

        plot_and_save(generated, output_path)

        return jsonify({
            "status": "success",
            "img_path": f"/static/{output_filename}"
        })
    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"status": "error", "message": str(e)})

# --- Start Server ---
if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    app.run(debug=True)
