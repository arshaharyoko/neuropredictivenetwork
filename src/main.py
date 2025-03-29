from pylsl import StreamInlet, resolve_streams
import numpy as np
import matplotlib.pyplot as plt
from npn import NPN

import time

streams = resolve_streams()
inlet = StreamInlet(streams[0])
num_channels = 9
channel_labels = ["Electrode", "Delta", "Theta", "Low Alpha", "High Alpha",
                  "Low Beta", "High Beta", "Low Gamma", "Mid Gamma"]

sampling_freq = 512
buf_size = 2048
chunk_size = 32
data = np.zeros((buf_size, num_channels))

plt.ion()
fig, axs = plt.subplots(num_channels, 1, figsize=(10, 12), sharex=True)
time_axis = np.linspace(0, buf_size/sampling_freq, buf_size)

lines = []
for ch in range(num_channels):
    line, = axs[ch].plot(time_axis, data[:, ch], 'k')
    axs[ch].set_ylabel(channel_labels[ch])
    lines.append(line)

axs[-1].set_xlabel('Time (s)')
fig.suptitle('EEG Signal')

npn = NPN(n_components_gmm=5, n_components_hmm=5, n_trials_hmm=1,
            wavelet='db6', level=1, random_state=0)

states = ["NEUTRAL", "UP", "DOWN", "LEFT", "RIGHT"]
calibration_duration = 10  # seconds per state
calibration_dict = {k: [] for k in range(len(states))}

def get_complete_chunk(inlet, chunk_size):
    buffer = []
    total = 0
    while total < chunk_size:
        chunk, t = inlet.pull_chunk(max_samples=chunk_size-total)
        if t:
            chunk = np.array(chunk)
            buffer.append(chunk)
            total += chunk.shape[0]
        else:
            time.sleep(0.005)

    full_chunk = np.concatenate(buffer, axis=0)
    return full_chunk[:chunk_size]

print("Starting calibration...")
for k, state in enumerate(states):
    input(f"\nPress Enter to start calibration for '{state}' ({calibration_duration} sec)")
    start = time.time()

    while time.time()-start < calibration_duration:
        chunk = get_complete_chunk(inlet, chunk_size)
        calibration_dict[k].append(chunk[:, 0])
        remaining = int(calibration_duration - (time.time() - start))
        print(f"THINK OF {state}: {remaining} sec remaining", end="\r")

    print(f"\nFinished calibration for '{state}'.")

npn.calibrate(calibration_dict)

while True:
    chunk = get_complete_chunk(inlet, chunk_size)
    n_samples = chunk.shape[0]

    # Display data prototype
    data = np.roll(data, -n_samples, axis=0)
    data[-n_samples:,:] = chunk[-n_samples:,:]
    
    # Predict rolling wave
    features, gmm_probabilities, gmm_labels, hidden_states = npn.process(chunk[:,0])
    print(gmm_probabilities)
    print(hidden_states)

    for ch in range(num_channels):
        lines[ch].set_ydata(data[:,ch])
        axs[ch].relim()
        axs[ch].autoscale_view()
    
    plt.pause(0.1)