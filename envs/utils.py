import os
import glob

def setup_training_dir(resume_training, algo, task, version):
    training_numbers = [int(folder.split("training")[-1]) for folder in glob.glob(f"training/{algo}/{task}/{version}/*")]
    if training_numbers == []: training_numbers = [0]
    if resume_training:
        training_number = max(training_numbers)
    else:
        training_number = max(training_numbers) + 1 if len(training_numbers) > 0 else 1
    os.makedirs(f"training/{algo}/{task}/{version}/training{training_number}", exist_ok=True)
    return training_number