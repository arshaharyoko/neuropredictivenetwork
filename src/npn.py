import numpy as np
import pywt
from gmm import GMM
from hmm import HMM
import json

class NPN:
    def __init__(self, n_components_gmm, n_components_hmm, n_trials_hmm, wavelet, level, buffer_size=32, random_state=0):
        self.n_components_gmm = n_components_gmm
        self.n_components_hmm = n_components_hmm
        self.n_trials_hmm = n_trials_hmm
        self.wavelet = wavelet
        self.level = level
        self.random_state = random_state
        
        self.buffer_size = buffer_size
        self.buffer = None

        self.gmm = GMM(n_components=self.n_components_gmm, random_state=self.random_state)
        self.hmm = HMM(n_components=self.n_components_hmm, n_trials=self.n_trials_hmm, n_iter=buffer_size, random_state=self.random_state)

    def label_to_onehot(self, label, n_categories):
        onehot = np.zeros(n_categories, dtype=int)
        onehot[label] = 1
        return onehot

    def dwt_extract_features(self, W):
        coeffs = pywt.wavedec(W, self.wavelet, level=self.level)
        # Use the approx. coefficients at the highest level as features
        return coeffs[0]
    
    def calibrate(self, W_dict):
        """
        Parameters:
            W_dict : state dictionary of sequences with len(buffer_size)
        """
        # 1. Figure out maximum length
        features_dict = {}
        for k, v in W_dict.items():
            features_dict[k] = []
            for W in v:
                features = self.dwt_extract_features(W)
                features_dict[k].append(features)
        
        features_dict_list = {k: [v.tolist() for v in vals] for k, vals in features_dict.items()}

        with open('features_dict.txt', 'w') as f:
            json.dump(features_dict_list, f, indent=2)

        # 2. Initialize and fit GMM
        means = [] 
        covariances = []
        weights = []
        total_samples = sum(len(v) for v in features_dict.values())
        for k in sorted(features_dict.keys()):
            X = np.vstack(features_dict[k])
            means.append(np.mean(X, axis=0))
            covariances.append(np.cov(X, rowvar=False))
            weights.append(X.shape[0]/total_samples)

        self.gmm.model.means_ = means
        self.gmm.model.covariances_ = covariances
        self.gmm.model.weights_ = weights

        X_train = np.concatenate([np.vstack(features_dict[k]) for k in sorted(features_dict.keys())], axis=0)
        self.gmm.fit(X_train)
        gmm_labels = self.gmm.predict(X_train)
        onehot_sequence = np.array([self.label_to_onehot(label, self.n_components_gmm) for label in gmm_labels])

        # 3. Initialize and fit HMM
        self.hmm.model.startprob_ = np.ones(self.n_components_hmm)
        self.hmm.model.transmat_ = np.full((self.n_components_hmm, self.n_components_hmm), 0.1) / self.n_components_gmm
        np.fill_diagonal(self.hmm.model.transmat_, 0.9)
        self.hmm.model.emissionprob_ = np.ones((self.n_components_hmm, self.n_components_gmm)) / self.n_components_gmm
        self.hmm.fit(onehot_sequence)

    def process(self, W):
        """
        Parameters:
            X: numpy array of len(buffer_size)
        Returns:
            tuple (features, gmm_probabilities, gmm_labels, hidden_states)
        """
        # 1. Preprocess data
        features = self.dwt_extract_features(W)

        if self.buffer is None or self.buffer.shape[1] != features.shape[0]:
            self.buffer = np.tile(features, (self.buffer_size, 1))
        else:
            self.buffer = np.roll(self.buffer, -1, axis=0)
            self.buffer[-1] = features

        # 2. Classification
        gmm_labels = self.gmm.predict(self.buffer)  
        gmm_probabilities = self.gmm.probability(self.buffer)
        onehot_sequence = np.array([self.label_to_onehot(label, self.n_components_gmm) for label in gmm_labels])
        
        # 3. Transitions
        hidden_states = self.hmm.predict(onehot_sequence)
        
        return features, gmm_probabilities, gmm_labels, hidden_states