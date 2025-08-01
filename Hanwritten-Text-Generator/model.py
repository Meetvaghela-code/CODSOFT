import numpy as np
import tensorflow as tf

# Load NPZ data
data = np.load("deepwriting_training.npz", allow_pickle=True)
strokes = data['strokes']
mean, std = data['mean'], data['std']

# Normalize strokes
def normalize_stroke(seq):
    return (seq - mean) / std

normalized_strokes = [normalize_stroke(s).astype(np.float32) for s in strokes]

# Pad sequences to match batch shape
def prepare_dataset(sequences, batch_size=64):
    def gen():
        for seq in sequences:
            input_seq = seq[:-1]
            target_seq = seq[1:]
            yield input_seq, target_seq

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32)
        )
    )

    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=([None, 3], [None, 3]),
        padding_values=(0.0, 0.0)
    )

    return dataset.shuffle(1000).prefetch(tf.data.AUTOTUNE)


train_dataset = prepare_dataset(normalized_strokes)

from tensorflow.keras import layers, models

def build_model(hidden_size=256, num_layers=2):
    model = models.Sequential()
    model.add(layers.Masking(mask_value=0.0, input_shape=(None, 3)))
    for _ in range(num_layers):
        model.add(layers.LSTM(hidden_size, return_sequences=True))
    model.add(layers.Dense(3))  # Predict Δx, Δy, pen_state
    return model

model = build_model()
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(train_dataset, epochs=20)
model.save('handwriting_rnn_tf.h5')

def generate_sequence(model, seed, length=700):
    sequence = []
    input_seq = tf.convert_to_tensor([seed], dtype=tf.float32)  # shape: (1, T, 3)

    for _ in range(length):
        prediction = model(input_seq, training=False)
        next_point = prediction[:, -1, :].numpy()[0]  # last timestep
        sequence.append(next_point)

        # Update input
        next_input = tf.expand_dims([[next_point]], axis=0)
        input_seq = tf.concat([input_seq, next_input[:, 0]], axis=1)

    return np.array(sequence)

seed = normalized_strokes[0][:10]  # Use first 10 points from a real stroke
generated = generate_sequence(model, seed)

import matplotlib.pyplot as plt

def render_strokes(seq, mean, std):
    seq = seq * std + mean  # Denormalize
    x, y = 0, 0
    X, Y = [], []

    for dx, dy, pen in seq:
        x += dx
        y += dy
        if pen < 0.5:  # pen down
            X.append(x)
            Y.append(y)
        else:
            if X:
                plt.plot(X, Y, 'k-', linewidth=1)
            X, Y = [], []

    if X:
        plt.plot(X, Y, 'k-', linewidth=1)

    plt.axis('equal')
    plt.axis('off')
    plt.gca().invert_yaxis()
    plt.show()

render_strokes(generated, mean, std)
