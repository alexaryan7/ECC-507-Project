import numpy as np
import matplotlib.pyplot as plt

# Config
N_SUB_CARRIERS = 1024
N_SYMBOLS_PER_SNR = 200
SNR_DB_RANGE = np.arange(0, 21, 2)

# QPSK
CONST_MAP = {
    (0, 0): -1 - 1j,
    (0, 1): -1 + 1j,
    (1, 1): 1 + 1j,
    (1, 0): 1 - 1j
}
INV_CONST_MAP = {v: k for k, v in CONST_MAP.items()}
CONST_POINTS = np.array(list(INV_CONST_MAP.keys()))
BITS_PER_SYMBOL = 2

# Channel
def create_thz_channel(N):
    channel_response = np.ones(N, dtype=complex)
    
    # Dips
    channel_response[100:150] = 0.001
    channel_response[400:425] = 0.001
    channel_response[700:800] = 0.001
    
    # Ripples
    channel_response[0:100] *= (0.8 + 0.2 * np.sin(np.linspace(0, 2*np.pi, 100)))
    channel_response[150:400] *= (0.9 + 0.1 * np.cos(np.linspace(0, np.pi, 250)))
    
    return channel_response

# Helpers
def qpsk_modulate(bits):
    symbols = []
    num_symbols = len(bits) // 2
    for i in range(num_symbols):
        b1 = bits[2*i]
        b2 = bits[2*i + 1]
        symbols.append(CONST_MAP[(b1, b2)])
    return np.array(symbols)

def qpsk_demodulate(received_symbols):
    demod_bits = []
    for sym in received_symbols:
        # Nearest
        distances = np.abs(CONST_POINTS - sym)
        nearest_point = CONST_POINTS[np.argmin(distances)]
        
        # Map
        bits = INV_CONST_MAP[nearest_point]
        demod_bits.extend(bits)
    return np.array(demod_bits)

# Plot Channel
thz_channel = create_thz_channel(N_SUB_CARRIERS)

plt.figure(figsize=(10, 5))
plt.plot(np.abs(thz_channel), label='Channel Gain')
plt.title('Simulated THz Channel Frequency Response')
plt.xlabel('Frequency Sub-carrier Index')
plt.ylabel('Magnitude (Gain)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('thz_channel_response.png')
print("Saved 'thz_channel_response.png'")

# Simulation
print("Starting simulation...")
ber_single_carrier = []
ber_adaptive_ofdm = []

# Adaptive
ADAPTIVE_THRESHOLD = 0.1
good_carriers = np.where(np.abs(thz_channel) > ADAPTIVE_THRESHOLD)[0]
n_good_carriers = len(good_carriers)
print(f"Adaptive OFDM will use {n_good_carriers} out of {N_SUB_CARRIERS} subcarriers.")

for snr_db in SNR_DB_RANGE:
    snr_linear = 10**(snr_db / 10)
    
    # Noise
    noise_variance_per_bin = 1.0 / snr_linear
    noise_std_dev = np.sqrt(noise_variance_per_bin / 2) 
    
    total_errors_sc = 0
    total_bits_sc = 0
    total_errors_ofdm = 0
    total_bits_ofdm = 0

    for _ in range(N_SYMBOLS_PER_SNR):
        
        # --- System 1: Naive SC ---
        sc_bits = np.random.randint(0, 2, N_SUB_CARRIERS * BITS_PER_SYMBOL)
        sc_symbols = qpsk_modulate(sc_bits)
        sc_symbols_normalized = sc_symbols / np.sqrt(2) 
        received_sc_symbols = sc_symbols_normalized * thz_channel
        noise = (np.random.normal(0, noise_std_dev, N_SUB_CARRIERS) + 
                 1j * np.random.normal(0, noise_std_dev, N_SUB_CARRIERS))
        received_sc_noisy = received_sc_symbols + noise
        demod_sc_bits = qpsk_demodulate(received_sc_noisy * np.sqrt(2))
        
        # Errors
        total_errors_sc += np.sum(sc_bits != demod_sc_bits)
        total_bits_sc += len(sc_bits)

        
        # --- System 2: Adaptive OFDM ---
        ofdm_bits = np.random.randint(0, 2, n_good_carriers * BITS_PER_SYMBOL)
        ofdm_symbols = qpsk_modulate(ofdm_bits)
        ofdm_symbols_normalized = ofdm_symbols / np.sqrt(2)
        
        # Load
        full_ofdm_symbol = np.zeros(N_SUB_CARRIERS, dtype=complex)
        full_ofdm_symbol[good_carriers] = ofdm_symbols_normalized
        
        # Channel
        received_ofdm_symbols_freq = full_ofdm_symbol * thz_channel
        noise = (np.random.normal(0, noise_std_dev, N_SUB_CARRIERS) + 
                 1j * np.random.normal(0, noise_std_dev, N_SUB_CARRIERS))
        received_ofdm_noisy = received_ofdm_symbols_freq + noise
        
        # Receiver
        symbols_to_demod = received_ofdm_noisy[good_carriers]
        
        # Equalize
        equalized_symbols = symbols_to_demod / thz_channel[good_carriers]
        
        # Demod
        demod_ofdm_bits = qpsk_demodulate(equalized_symbols * np.sqrt(2))
        
        # Errors
        total_errors_ofdm += np.sum(ofdm_bits != demod_ofdm_bits)
        total_bits_ofdm += len(ofdm_bits)

    # BER
    ber_sc = total_errors_sc / total_bits_sc if total_bits_sc > 0 else 0
    ber_ofdm = total_errors_ofdm / total_bits_ofdm if total_bits_ofdm > 0 else 0
    
    ber_single_carrier.append(ber_sc)
    ber_adaptive_ofdm.append(ber_ofdm)
    print(f"SNR: {snr_db} dB | BER (SC): {ber_sc:.2e} | BER (OFDM): {ber_ofdm:.2e}")

# Plot BER
plt.figure(figsize=(10, 6))
plt.plot(SNR_DB_RANGE, ber_single_carrier, 'bo-', label='Naive Single-Carrier (Matched Filter)')
plt.plot(SNR_DB_RANGE, ber_adaptive_ofdm, 'gs-', label='Adaptive OFDM Receiver')
plt.yscale('log')
plt.ylim(1e-5, 1.0)
plt.xlabel('Signal-to-Noise Ratio (SNR) in dB')
plt.ylabel('Bit Error Rate (BER)')
plt.title('Performance of Naive vs. Adaptive Receiver in THz Channel')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('thz_ber_comparison.png')
print("Saved 'thz_ber_comparison.png'")

print("\nSimulation complete. All plots saved.")