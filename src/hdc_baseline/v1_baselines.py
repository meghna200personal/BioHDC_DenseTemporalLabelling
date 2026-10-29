import numpy as np 

#should be replaced with actual data loading and preprocessing functions
def generate_dummy_gal_data(num_samples=1500, num_channels=32, num_events=6, seed=42): 
    """ Generates synthetic 32-channel continuous EEG signals and 6 binary event target channels. """ 
    np.random.seed(seed) 
    time = np.linspace(0, 10, num_samples) 
    eeg_data = np.zeros((num_samples, num_channels)) 
    events = np.zeros((num_samples, num_events), dtype=int) 
    # 1. Generate active event windows across samples 
    for k in range(num_events): 
        period = 250 
        for cycle in range(num_samples // period): 
            start = cycle * period + k * 30 
            end = min(start + 35, num_samples) 
            events[start:end, k] = 1 
    # 2. Inject spatial-frequency EEG signatures into specific channels during active events 
    for c in range(num_channels): 
        signal = np.sin(2 * np.pi * (5 + (c % 4)) * time) + 0.5 * np.cos(2 * np.pi * 15 * time) 
        for k in range(num_events): 
            if c % num_events == k: 
                signal += 3.0 * events[:, k] * np.sin(2 * np.pi * (10 + k * 8) * time) 
                eeg_data[:, c] = signal + np.random.normal(0, 0.2, num_samples) 
    return eeg_data, events 
    
class BinaryHDCEncoder: 
    def __init__(self, D=10000, num_channels=32, num_levels=21, seed=42): 
        self.D = D 
        self.num_channels = num_channels 
        self.num_levels = num_levels 
        np.random.seed(seed) 
        # 1. Channel Item Memory (Orthogonal random binary hypervectors) 
        self.IM = np.random.randint(0, 2, size=(num_channels, D), dtype=np.uint8) 
        # 2. Continuous Item Memory (CIM) with similarity preservation across levels 
        self.CIM = np.zeros((num_levels, D), dtype=np.uint8) 
        base_vector = np.random.randint(0, 2, size=D, dtype=np.uint8) 
        self.CIM[0] = base_vector 
        bits_to_flip = np.random.permutation(D) 
        step = D // (2 * (num_levels - 1)) 
        for m in range(1, num_levels): 
            level_vector = self.CIM[m-1].copy() 
            flip_indices = bits_to_flip[(m-1)*step : m*step] 
            level_vector[flip_indices] = 1 - level_vector[flip_indices] 
            self.CIM[m] = level_vector 
    def quantize_and_map(self, eeg_sample, min_val=-4.0, max_val=4.0): 
        """Quantizes continuous 32-channel voltages into CIM level indices.""" 
        clipped = np.clip(eeg_sample, min_val, max_val) 
        normalized = (clipped - min_val) / (max_val - min_val) 
        level_indices = np.floor(normalized * (self.num_levels - 1)).astype(int) 
        return self.CIM[level_indices] # Shape: (32, D) 
    def encode_spatial_snapshot(self, eeg_sample): 
        """ Baseline 1: S_t = threshold( sum( E_c XOR V_t^c ) ) """ 
        level_vectors = self.quantize_and_map(eeg_sample) # Bitwise XOR binding between electrode ID and signal level 
        bound_channels = np.bitwise_xor(self.IM, level_vectors) 
        # Majority rule bundling across 32 channels 
        summed = np.sum(bound_channels, axis=0) 
        spatial_snapshot = (summed > (self.num_channels / 2)).astype(np.uint8) 
        return spatial_snapshot 
    def encode_spatial_temporal_ngram(self, eeg_window, n_gram=10): 
        """ Baseline 2 (Rahimi et al.): T_t = threshold( sum( rho^i( S_{t-i} ) ) ) """ 
        permuted_snapshots = [] 
        for i in range(n_gram): 
            sample_at_lag = eeg_window[-(i+1)] 
            S_lag = self.encode_spatial_snapshot(sample_at_lag) 
            # Cyclic shift permutation rho^i 
            rho_S_lag = np.roll(S_lag, shift=i) 
            permuted_snapshots.append(rho_S_lag) 
        summed = np.sum(permuted_snapshots, axis=0) 
        ngram_vector = (summed > (n_gram / 2)).astype(np.uint8) 
        return ngram_vector 
    def compute_hamming_similarity(query, prototype): 
        """Normalized Hamming Similarity: 1.0 - (Hamming Distance / D)""" 
        return 1.0 - (np.sum(query != prototype) / len(query)) 
    def main(): 
        D = 10000 
        num_samples = 1500 
        num_channels = 32 
        num_events = 6 
        n_gram = 10 
        eeg_data, events = generate_dummy_gal_data(num_samples=num_samples, num_channels=num_channels, num_events=num_events) 
        # Train / Test Split 
        train_split = 1000 
        train_eeg, test_eeg = eeg_data[:train_split], eeg_data[train_split:] 
        train_y, test_y = events[:train_split], events[train_split:] 
        hdc = BinaryHDCEncoder(D=D, num_channels=num_channels, num_levels=21) 
        # ------------------------------------------------------------------------- # BASELINE 1: Spatial-Only Instantaneous Frame Encoder # ------------------------------------------------------------------------- 
        prototypes_b1_accum = np.zeros((num_events, D), dtype=int) 
        for t in range(train_split): 
            S_t = hdc.encode_spatial_snapshot(train_eeg[t]) 
            bipolar_S_t = np.where(S_t == 1, 1, -1) 
            for k in range(num_events): 
                if train_y[t, k] == 1: 
                    prototypes_b1_accum[k] += bipolar_S_t 
                else: 
                    prototypes_b1_accum[k] -= bipolar_S_t 
        prototypes_b1 = (prototypes_b1_accum > 0).astype(np.uint8) 
        sims_b1 = np.zeros((len(test_eeg), num_events)) 
        for t in range(len(test_eeg)): 
            S_t = hdc.encode_spatial_snapshot(test_eeg[t]) 
            for k in range(num_events): 
                sims_b1[t, k] = compute_hamming_similarity(S_t, prototypes_b1[k]) 
        # ------------------------------------------------------------------------- # BASELINE 2: Rahimi-Style Spatial-Temporal n-gram Encoder # ------------------------------------------------------------------------- 
        prototypes_b2_accum = np.zeros((num_events, D), dtype=int) 
        for t in range(n_gram - 1, train_split): 
            window = train_eeg[t - n_gram + 1 : t + 1] 
            T_t = hdc.encode_spatial_temporal_ngram(window, n_gram=n_gram) 
            bipolar_T_t = np.where(T_t == 1, 1, -1) 
            for k in range(num_events): 
                if train_y[t, k] == 1: 
                    prototypes_b2_accum[k] += bipolar_T_t 
                else: 
                    prototypes_b2_accum[k] -= bipolar_T_t 
        prototypes_b2 = (prototypes_b2_accum > 0).astype(np.uint8) 
        sims_b2 = np.zeros((len(test_eeg) - n_gram + 1, num_events)) 
        test_labels_b2 = test_y[n_gram - 1 :] 
        for idx, t in enumerate(range(n_gram - 1, len(test_eeg))): 
            window = test_eeg[t - n_gram + 1 : t + 1] 
            T_t = hdc.encode_spatial_temporal_ngram(window, n_gram=n_gram) 
            for k in range(num_events): 
                sims_b2[idx, k] = compute_hamming_similarity(T_t, prototypes_b2[k]) 
        # ------------------------------------------------------------------------- # Print Summary & Metrics # ------------------------------------------------------------------------- 
        event_names = ["HandStart", "FirstDigitTouch", "BothStartLoadPhase", "LiftOff", "Replace", "BothRelease"] 
        print("========================================================================") 
        print(" HDC BASELINE VALIDATION SUMMARY (GAL)") 
        print("========================================================================") 
        print(f"Hypervector Dimension (D): {D} bits") 
        print(f"Number of Spatial Channels: {num_channels}") 
        print(f"Signal Level Granularity (CIM): {21} levels") 
        print(f"Memory Footprint of 6 Binary Event Prototypes: {6 * D // 8} Bytes ({6 * D / 8000:.2f} KB)") 
        print("\\n--- Baseline 1: Spatial-Only Encoder ---") 
        for k in range(num_events): 
            on_mask = test_y[:, k] == 1 
            off_mask = test_y[:, k] == 0 
            sim_on = np.mean(sims_b1[on_mask, k]) 
            sim_off = np.mean(sims_b1[off_mask, k]) 
            print(f" Event {k+1} ({event_names[k]:<18}): ON_Sim = {sim_on:.4f} | OFF_Sim = {sim_off:.4f} | Margin = {sim_on - sim_off:+.4f}") 
        print("\\n--- Baseline 2: Spatial-Temporal n-gram Encoder (n = 10) ---") 
        for k in range(num_events): 
            on_mask = test_labels_b2[:, k] == 1 
            off_mask = test_labels_b2[:, k] == 0 
            sim_on = np.mean(sims_b2[:, k]) 
            sim_off = np.mean(sims_b2[:, k]) 
            print(f" Event {k+1} ({event_names[k]:<18}): ON_Sim = {sim_on:.4f} | OFF_Sim = {sim_off:.4f} | Margin = {sim_on - sim_off:+.4f}")