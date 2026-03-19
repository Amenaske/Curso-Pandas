import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import hls4ml

# Set seeds for reproducibility
np.random.seed(5)
tf.random.set_seed(5)

## --- Parameters ---
N = 8
M = 8
n_iteration = 1
n_epoch = 1000
modulation_order = 2
SNR_db = np.array(list(range(0, 11, 2)))
batch_size = 5
test_size = 1000


## --- Data Generation Functions ---
def generate_data(M, batch_size):
    input_ = np.zeros((M, 1 << M))
    label = np.zeros((2 * M, 1 << M))

    for i in range(1 << M):
        for j in range(M):
            if i & (1 << j):
                input_[M - j - 1][i] = 1

    for i in range(1 << M):
        for j in range(M):
            if input_[j][i] == 1:
                label[2 * j][i] = 1
                label[2 * j + 1][i] = 0
            else:
                label[2 * j][i] = 0
                label[2 * j + 1][i] = 1

    input_tile = np.tile(input_, batch_size)
    label_tile = np.tile(label, batch_size)
    return input_tile, label_tile


def generate(M, N, batch_size):
    data, label = generate_data(M, batch_size)

    ran1 = np.random.permutation(data.shape[1])
    ran2 = np.random.permutation(data.shape[1])

    symbol1 = 2 * data[:, ran1] - 1
    symbol2 = 2 * data[:, ran2] - 1

    # Superposition Coding (NOMA)
    SPC = math.sqrt(0.8) * symbol1 + math.sqrt(0.2) * symbol2

    label1 = np.transpose(label[:, ran1].astype('float32'))
    label2 = np.transpose(label[:, ran2].astype('float32'))

    return SPC, label1, label2


def generate_channel(N, M, k):
    h_real = np.random.randn(N, M) / math.sqrt(2)
    h_imag = np.random.randn(N, M) / math.sqrt(2)
    if k == 0:
        return h_real, h_imag
    else:
        return 2 * h_real, 2 * h_imag


def generate_input(H_real, H_imag, SPC, N, batch_size, sigma):
    # Noise generation
    N_real = np.random.randn(N, SPC.shape[1]) / math.sqrt(2)
    N_imag = np.random.randn(N, SPC.shape[1]) / math.sqrt(2)

    input_real = np.matmul(H_real, SPC) + sigma * N_real
    input_imag = np.matmul(H_imag, SPC) + sigma * N_imag

    # Concatenate real and imaginary parts for the DNN input
    return np.transpose(np.concatenate((input_real, input_imag), axis=0)).astype('float32')


## --- Manual Accuracy Calculation ---
def calculate_manual_accuracy(y_true, y_pred, M):
    """
    Calculates accuracy by comparing the max logit of each bit pair.
    """
    correct_bits = 0
    total_bits = y_true.shape[0] * M

    for i in range(M):
        # Slice out the pair of bits (2 columns per symbol)
        true_slice = y_true[:, 2 * i: 2 * i + 2]
        pred_slice = y_pred[:, 2 * i: 2 * i + 2]

        # Get index of max (0 or 1)
        true_idx = np.argmax(true_slice, axis=1)
        pred_idx = np.argmax(pred_slice, axis=1)

        correct_bits += np.sum(true_idx == pred_idx)

    return correct_bits / total_bits


## --- Define the Receiver Model ---
def ReceiverModel(modulation_order, N):
    # Define the input layer explicitly
    inputs = tf.keras.Input(shape=(2 * N,), name="input_layer")

    # Define the layers exactly like your previous architecture
    x = tf.keras.layers.BatchNormalization()(inputs)

    x = tf.keras.layers.Dense(16 * modulation_order, activation='relu',
                              kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(16 * modulation_order, activation='relu',
                              kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(16 * modulation_order, activation='relu',
                              kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(16 * modulation_order, activation='relu',
                              kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Output layer
    outputs = tf.keras.layers.Dense(modulation_order * N)(x)
    outputs = tf.keras.layers.BatchNormalization()(outputs)

    # Construct the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# Instantiate them
model1 = ReceiverModel(modulation_order, N)
model2 = ReceiverModel(modulation_order, N)

# Initialize models and optimizers
model1 = ReceiverModel(modulation_order, N)
model2 = ReceiverModel(modulation_order, N)
optimizer1 = tf.keras.optimizers.Adam(learning_rate=1e-4)
optimizer2 = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

## --- Training Loop ---
ERROR_user1 = np.zeros([len(SNR_db), n_iteration])
ERROR_user2 = np.zeros([len(SNR_db), n_iteration])

for k in range(n_iteration):
    print(f'\n--- Training Iteration {k} ---')
    H1_real, H1_imag = generate_channel(N, M, 0)
    H2_real, H2_imag = generate_channel(N, M, 1)

    # Train User 1
    for i, snr in enumerate(SNR_db):
        print(f'User 1, SNR: {snr}')
        for j in range(n_epoch):
            SPC, label1, _ = generate(M, N, batch_size)
            sig_pwr = np.mean(SPC ** 2)
            sigma = math.sqrt(sig_pwr / (math.sqrt(N) * 10 ** (snr / 10.0)))
            train_in = generate_input(H1_real, H1_imag, SPC, N, batch_size, sigma)

            with tf.GradientTape() as tape:
                logits = model1(train_in, training=True)
                # Reshape to treat each symbol's pair of logits as a separate distribution for the loss
                # Or just use the flattened version if matches TF1 behavior
                loss_val = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label1, logits=logits))

            grads = tape.gradient(loss_val, model1.trainable_variables)
            optimizer1.apply_gradients(zip(grads, model1.trainable_variables))

            if j % 200 == 0:
                acc = calculate_manual_accuracy(label1, logits.numpy(), M)
                print(f" Epoch {j}: Loss {loss_val:.4f}, Accuracy {acc:.4f}")

    # Train User 2
    for i, snr in enumerate(SNR_db):
        print(f'User 2, SNR: {snr}')
        for j in range(n_epoch):
            SPC, _, label2 = generate(M, N, batch_size)
            sig_pwr = np.mean(SPC ** 2)
            sigma = math.sqrt(sig_pwr / (math.sqrt(N) * 10 ** (snr / 10.0)))
            train_in = generate_input(H2_real, H2_imag, SPC, N, batch_size, sigma)

            with tf.GradientTape() as tape:
                logits = model2(train_in, training=True)
                loss_val = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label2, logits=logits))

            grads = tape.gradient(loss_val, model2.trainable_variables)
            optimizer2.apply_gradients(zip(grads, model2.trainable_variables))

            if j % 200 == 0:
                acc = calculate_manual_accuracy(label2, logits.numpy(), M)
                print(f" Epoch {j}: Loss {loss_val:.4f}, Accuracy {acc:.4f}")

    ## --- Testing ---
    print(f'Testing iteration {k}...')
    for i, snr in enumerate(SNR_db):
        current_snr_db = SNR_db[i]
        SPC_t, label1_t, label2_t = generate(M, N, batch_size * test_size)
        sig_pwr = np.mean(SPC_t ** 2)
        sigma = math.sqrt(sig_pwr / (math.sqrt(N) * 10 ** (snr / 10.0)))

        # User 1 Test
        in1_t = generate_input(H1_real, H1_imag, SPC_t, N, batch_size * test_size, sigma)
        out1_t = model1(in1_t, training=False).numpy()
        ERROR_user1[i, k] = 1 - calculate_manual_accuracy(label1_t, out1_t, M)

        # User 2 Test
        in2_t = generate_input(H2_real, H2_imag, SPC_t, N, batch_size * test_size, sigma)
        out2_t = model2(in2_t, training=False).numpy()
        ERROR_user2[i, k] = 1 - calculate_manual_accuracy(label2_t, out2_t, M)

        print(f"SNR {current_snr_db} dB: SER1 = {ERROR_user1[i, k]:.6f}, SER2 = {ERROR_user2[i, k]:.6f}")

config = hls4ml.utils.config_from_keras_model(model1)
hls_model = hls4ml.converters.convert_from_keras_model(
   model=model1,
   hls_config=config,
   backend='Vitis'
)
hls_model.build()
hls4ml.report.read_vivado_report('my-hls-test')

## --- Plotting ---
error1 = np.mean(ERROR_user1, axis=1)
error2 = np.mean(ERROR_user2, axis=1)

plt.figure()
plt.semilogy(SNR_db, error1, ls='--', marker='o', label='User 1')
plt.semilogy(SNR_db, error2, ls='--', marker='+', label='User 2')
plt.grid(True)
plt.legend()
plt.ylim(pow(10, -6), pow(10, 0))
plt.xlabel('SNR (dB)')
plt.ylabel('SER')
plt.title('SER of NOMA Users via DNN (TF2)')
plt.show()

print("User 1 SER:", error1)
print("User 2 SER:", error2)