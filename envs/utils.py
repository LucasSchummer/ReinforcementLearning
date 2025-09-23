import os
import glob
import numpy as np


class Normalizer:
    def __init__(self, size, eps=1e-8):
        self.size = size
        self.mean = np.zeros(size, dtype=np.float32)
        self.var = np.ones(size, dtype=np.float32)
        self.count = eps
        self.eps = eps

    def update(self, batch):
        """Update running mean/var with a batch"""
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        # Update mean
        new_mean = self.mean + delta * batch_count / tot_count

        # Update var (Welford’s trick)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        # Save
        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, batch):
        """Normalize goals using stored stats"""
        return (batch - self.mean) / (np.sqrt(self.var) + self.eps)


def setup_training_dir(resume_training, algo, task, version):
    training_numbers = [int(folder.split("training")[-1]) for folder in glob.glob(f"training/{algo}/{task}/{version}/*")]
    if training_numbers == []: training_numbers = [0]
    if resume_training:
        training_number = max(training_numbers)
    else:
        training_number = max(training_numbers) + 1 if len(training_numbers) > 0 else 1
    os.makedirs(f"training/{algo}/{task}/{version}/training{training_number}", exist_ok=True)
    return training_number