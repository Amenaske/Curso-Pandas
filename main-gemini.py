import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

# Set up environment for TF 2.x
tf.random.set_seed(5)  # Set TF global seed
np.random.seed(5)  # Set Numpy seed

# --- Parameters ---
N = 8  # Number of Receive Antennas (User side)
M = 8  # Number of Transmit Antennas (Base station side)
n_symbol = 1000  # (Original variable, but not used in the logic)
n_iteration = 1
n_epoch = 1000
modulation_order = 2  # BPSK outputs 2 labels per bit (0/1 and 1/0)
SNR_db = np.array(list(range(0, 11, 2)))
batch_size = 5
test_size = 1000

ERROR_user1 = np.zeros([len(SNR_db), n_iteration])
ERROR_user2 = np.zeros([len(SNR_db), n_iteration])


# --- Data Generation Functions ---

def generate_data(M, batch_size):
    """
    Generates BPSK data bits and one-hot labels for M bits.
    Each row of input_ is a bit stream (0 or 1).
    Each row of label is a one-hot representation (2 * M rows total).
    """
    input_ = []
    for i in range(1 << M):
        bit_config = [(i >> j) & 1 for j in range(M)]
        input_.append(bit_config[::-1])  # MSB first (index 0)

    input_ = np.array(input_).T

    # Generate one-hot labels (2M rows)
    # Bit 0 (0,1) or Bit 1 (1,0)
    label = np.zeros((2 * M, 1 << M))
    for i in range(1 << M):
        for j in range(M):
            if input_[j][i] == 1:
                label[2 * j][i] = 1.0  # Index 0 is 1 (bit 1)
                label[2 * j + 1][i] = 0.0  # Index 1 is 0 (bit 1)
            else:
                label[2 * j][i] = 0.0  # Index 0 is 0 (bit 0)
                label[2 * j + 1][i] = 1.0  # Index 1 is 1 (bit 0)

    # Tile to fill the batch
    input_ = np.tile(input_, batch_size)
    label = np.tile(label, batch_size)

    return input_, label


def generate(M, N, batch_size):
    """
    Generates Superposition Coded (SPC) symbols and corresponding labels.
    Uses power allocation: User1: 0.8, User2: 0.2 (stronger user is user1, decoded first).
    """
    data, label = generate_data(M, batch_size)

    total_symbols = batch_size * pow(2, M)
    ran1 = np.random.permutation(total_symbols)  # Suffling Dataset
    ran2 = np.random.permutation(total_symbols)

    # BPSK modulation: 0 -> -1, 1 -> 1
    symbol1 = 2 * data[:, ran1] - 1
    symbol2 = 2 * data[:, ran2] - 1

    # Superposition Coding (SPC)
    # The symbol dimension is M x (total_symbols)
    SPC = math.sqrt(0.8) * symbol1 + math.sqrt(0.2) * symbol2

    # Transpose labels for Keras/TF: (total_symbols) x (2*M)
    label1 = np.transpose(label[:, ran1].astype('float32'))
    label2 = np.transpose(label[:, ran2].astype('float32'))

    return SPC, label1, label2


def generate_channel(N, M, k):
    """
    Generates i.i.d. complex Gaussian channel matrix H (real and imaginary parts).
    k=0 for H1, k=1 for H2 (stronger channel).
    """
    # Scaling by 1/sqrt(2) for complex Gaussian with unit variance.
    h1 = np.random.randn(N, M) / math.sqrt(2)
    h2 = np.random.randn(N, M) / math.sqrt(2)

    if k == 0:
        return h1, h2
    else:
        # Channel H2 is 2*H1 in the original code
        return 2 * h1, 2 * h2


def generate_input(H_real, H_image, SPC, N, num_symbols, sigma):
    """
    Generates the noisy received signal: y = H * s + n.
    Input to the DNN is [Re(y); Im(y)].
    """
    # Noise: complex Gaussian with variance sigma^2 (i.e., real/imag variance sigma^2 / 2)
    # The original code's generate_channel function for noise (k=0) scales by 1/sqrt(2) already.
    # The noise in generate_channel(N, M, 0) is actually N x M for M symbols. We need N x num_symbols.

    N_real, N_image = np.random.randn(N, num_symbols) / math.sqrt(2), np.random.randn(N, num_symbols) / math.sqrt(2)

    # Received Signal Real/Imaginary parts: Re(y) = Re(H*s) + Re(n), Im(y) = Im(H*s) + Im(n)
    # H = H_real + j * H_image, s is real (BPSK is real symbol)
    # y = (H_real + j * H_image) * s + sigma * (N_real + j * N_image)
    # Re(y) = H_real * s + sigma * N_real
    # Im(y) = H_image * s + sigma * N_image

    input_real = np.matmul(H_real, SPC) + sigma * N_real
    input_img = np.matmul(H_image, SPC) + sigma * N_image

    # Concatenate [Re(y); Im(y)] along the antenna dimension (axis=0). Dimension: (2*N) x (num_symbols)
    # Transpose for Keras/TF: (num_symbols) x (2*N)
    input_data = np.transpose(np.concatenate((input_real, input_img), axis=0))

    return input_data


# --- Keras Model Definition ---

def reciever_model(num):
    """
    Defines the DNN receiver using tf.keras.Sequential.
    The architecture is: BN -> Dense(ReLU) -> BN -> ... -> Dense(linear) -> BN
    """
    # Use Glorot uniform initializer as the standard in Keras, replacing truncated_normal(stddev=0.01)
    initializer = tf.keras.initializers.GlorotUniform()

    # Layer definitions
    model = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(input_shape=(2 * N,), name=f'rx{num}_bn0'),
        tf.keras.layers.Dense(16 * modulation_order, activation='relu', kernel_initializer=initializer,
                              name=f'rx{num}_dense1'),
        tf.keras.layers.BatchNormalization(name=f'rx{num}_bn1'),
        tf.keras.layers.Dense(16 * modulation_order, activation='relu', kernel_initializer=initializer,
                              name=f'rx{num}_dense2'),
        tf.keras.layers.BatchNormalization(name=f'rx{num}_bn2'),
        tf.keras.layers.Dense(16 * modulation_order, activation='relu', kernel_initializer=initializer,
                              name=f'rx{num}_dense3'),
        tf.keras.layers.BatchNormalization(name=f'rx{num}_bn3'),
        tf.keras.layers.Dense(16 * modulation_order, activation='relu', kernel_initializer=initializer,
                              name=f'rx{num}_dense4'),
        tf.keras.layers.BatchNormalization(name=f'rx{num}_bn4'),
        tf.keras.layers.Dense(modulation_order * M, kernel_initializer=initializer, name=f'rx{num}_logit'),
        tf.keras.layers.BatchNormalization(name=f'rx{num}_bn_out')  # Output BN layer
    ], name=f'reciever_{num}')

    return model


# --- Custom Metric for Symbol Error Rate (SER) ---

class SymbolAccuracy(tf.keras.metrics.Metric):
    """
    Custom metric to calculate the Symbol Accuracy for the M BPSK symbols.
    It averages the bit accuracy for all M bits, which is equivalent to the
    Symbol Accuracy (SER = 1 - Accuracy).
    """

    def __init__(self, name='symbol_accuracy', M=M, **kwargs):
        super(SymbolAccuracy, self).__init__(name=name, **kwargs)
        self.M = M
        # Accumulates the sum of batch accuracies (a floating point number)
        self.total_acc = self.add_weight(name='total_acc', initializer='zeros', dtype=tf.float32)
        # Accumulates the number of batches processed
        self.num_batches = self.add_weight(name='num_batches', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        total_bit_acc = 0.0

        # Calculate the average bit accuracy (Symbol Accuracy) for the current batch
        for i in range(self.M):
            pred_slice = y_pred[:, 2 * i: 2 * i + 2]
            true_slice = y_true[:, 2 * i: 2 * i + 2]

            pred_class = tf.argmax(pred_slice, axis=1)
            true_class = tf.argmax(true_slice, axis=1)

            bit_acc = tf.reduce_mean(tf.cast(tf.equal(pred_class, true_class), tf.float32))
            total_bit_acc += bit_acc

        avg_bit_acc = total_bit_acc / self.M

        # 1. Add the batch accuracy to the total accumulator
        self.total_acc.assign_add(avg_bit_acc)
        # 2. Increment the batch count
        self.num_batches.assign_add(1.0)  # Always increments by 1 for a single batch step

    def result(self):
        # The result is the running mean of the batch accuracies.
        return self.total_acc / self.num_batches

    def reset_states(self):
        self.total_acc.assign(0.0)
        self.num_batches.assign(0.0)


# --- Model Initialization and Compilation ---

# Receiver for User 1 (Stronger channel/decoded first)
rx1_model = reciever_model(1)
rx1_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    # Use from_logits=True since the model output is not softmaxed
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[SymbolAccuracy(M=M)]
)

# Receiver for User 2 (Weaker channel/decoded second)
rx2_model = reciever_model(2)
rx2_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[SymbolAccuracy(M=M)]
)

# --- Training and Testing Loop ---

for k in range(n_iteration):
    print(f'*** Training iteration {k} ***')

    # Generate fixed channel matrices for this iteration
    H1_real, H1_image = generate_channel(N, M, 0)
    H2_real, H2_image = generate_channel(N, M, 1)

    # --- Training User 1 ---
    for i in range(len(SNR_db)):
        current_snr_db = SNR_db[i]
        print(f'--- Training User 1, SNR: {current_snr_db} dB ---')

        for j in range(n_epoch):
            # Generate training data batch
            SPC, label1_train, _ = generate(M, N, batch_size)
            signal_power = np.mean(pow(SPC, 2))
            # Calculate noise standard deviation (sigma)
            sigma_user1 = math.sqrt(signal_power / (math.sqrt(N) * pow(10, float(current_snr_db) / 10.0)))

            input1_train = generate_input(H1_real, H1_image, SPC, N, batch_size * pow(2, M), sigma_user1)

            # Train with one step
            log = rx1_model.train_on_batch(input1_train, label1_train)

            if j % 100 == 0:
                loss, accuracy = log[0], log[1]
                print(f'Epoch {j}: loss: {loss:.4f}, accuracy: {accuracy:.4f}')

        # Reset the receiver's state after training on this SNR to prepare for the next
        # (This mimics the original TF1 code structure where weights are kept, but the
        # loop structure suggests a new training phase for each SNR).
        # We will NOT reset weights, just continue training with new data.

    # --- Training User 2 ---
    for i in range(len(SNR_db)):
        current_snr_db = SNR_db[i]
        print(f'--- Training User 2, SNR: {current_snr_db} dB ---')

        for j in range(n_epoch):
            # Generate training data batch
            SPC, _, label2_train = generate(M, N, batch_size)
            signal_power = np.mean(pow(SPC, 2))
            # Calculate noise standard deviation (sigma)
            sigma_user2 = math.sqrt(signal_power / (math.sqrt(N) * pow(10, float(current_snr_db) / 10.0)))

            input2_train = generate_input(H2_real, H2_image, SPC, N, batch_size * pow(2, M), sigma_user2)

            # Train with one step
            log = rx2_model.train_on_batch(input2_train, label2_train)

            if j % 100 == 0:
                loss, accuracy = log[0], log[1]
                print(f'Epoch {j}: loss: {loss:.4f}, accuracy: {accuracy:.4f}')

# --- Testing Loop (Symbol Error Rate Calculation) ---

for k in range(n_iteration):
    print(f'*** Testing operation {k} ***')

    for i in range(len(SNR_db)):
        current_snr_db = SNR_db[i]

        # Generate test data (large batch)
        SPC_test, test_label1, test_label2 = generate(M, N, batch_size * test_size)
        signal_power = np.mean(pow(SPC_test, 2))  # Use SPC_test for power calculation
        sigma = math.sqrt(signal_power / (math.sqrt(N) * pow(10, float(current_snr_db) / 10.0)))

        num_test_symbols = batch_size * test_size * pow(2, M)

        # --- Test User 1 ---
        input1_test = generate_input(H1_real, H1_image, SPC_test, N, num_test_symbols, sigma)
        # Use evaluate for accuracy, predict for raw output if needed
        _, ac1 = rx1_model.evaluate(input1_test, test_label1, verbose=0)
        ERROR_user1[i, k] = 1 - ac1  # SER = 1 - Accuracy

        # --- Test User 2 ---
        input2_test = generate_input(H2_real, H2_image, SPC_test, N, num_test_symbols, sigma)
        _, ac2 = rx2_model.evaluate(input2_test, test_label2, verbose=0)
        ERROR_user2[i, k] = 1 - ac2  # SER = 1 - Accuracy

        print(f"SNR {current_snr_db} dB: SER1 = {ERROR_user1[i, k]:.6f}, SER2 = {ERROR_user2[i, k]:.6f}")

# --- Plotting Results ---

error1 = np.mean((ERROR_user1), axis=1)
error2 = np.mean((ERROR_user2), axis=1)

print("\n--- Final Results ---")
print(f"User 1 SER: {error1}")
print(f"User 2 SER: {error2}")
print(f"H1_real (sample): \n{H1_real[:, :4]}")  # Print a snippet of H1_real

plt.figure()
plt.semilogy(SNR_db, error1, ls='--', marker='o', label='user1')
plt.semilogy(SNR_db, error2, ls='--', marker='*', label='user2')
plt.grid(True)
plt.legend()
plt.ylim(pow(10, -6), pow(10, 0))
plt.xlabel('SNR (dB)')
plt.ylabel('SER (Symbol Error Rate)')
plt.title('SER of User 1 and User 2 in MIMO-NOMA BPSK-DNN')
plt.savefig('SER_44MIMO_NOMA_DNN_BPSK_TF2')
plt.show()