# Fingerprint and Iris Recognition with CNN

A multimodal biometric verification system. The user uploads a fingerprint
image and an iris image through a desktop interface, and the system decides
whether the two samples belong to the same enrolled person. Two separate
convolutional neural networks extract features from the fingerprint and the
iris, those features are matched against a stored database of enrolled people,
and the two results are combined into a single decision.


## Purpose

The goal of the project is to demonstrate how two different biometric traits
can be combined to verify a person's identity more reliably than either trait
alone. A single biometric can be noisy or ambiguous, so the system uses both
the fingerprint and the iris and fuses their scores. This is a common technique
in real biometric systems and is the core idea the project illustrates.


## How It Works

The system follows five stages.

1. Data. Real fingerprint and iris datasets require licences and large
   downloads, so the project generates its own synthetic dataset. Each subject
   is given a unique fingerprint pattern and a unique iris texture, and several
   slightly varied captures of each are produced so the networks can learn to
   recognise a person across different captures.

2. Feature extraction. Two convolutional neural networks are trained, one for
   fingerprints and one for irises. Each network learns to tell the enrolled
   subjects apart. The layer just before the final classification layer is used
   as a feature vector, or embedding, that represents the biometric.

3. Enrollment. Every capture of a subject is passed through the network and the
   resulting embeddings are averaged into one template per subject per modality.
   These templates form the stored database that new samples are compared
   against.

4. Matching. When a new fingerprint and iris are submitted, the system computes
   their embeddings and compares them against every stored template using cosine
   similarity, giving a best matching subject and a confidence score for each
   modality.

5. Fusion and decision. The fingerprint score and the iris score are combined
   with a weighted sum. Based on the combined score and whether both modalities
   point to the same person, the system returns one of three results:

   - Same person: both the fingerprint and the iris match the same enrolled
     subject with high confidence.
   - Different people: both samples are recognised, but they belong to two
     different enrolled subjects.
   - Not recognised: at least one sample cannot be confidently matched to anyone
     in the database.


## Technologies Used

- Python
- TensorFlow and Keras for the convolutional neural networks
- NumPy for numerical operations
- Pillow for image generation and processing
- CustomTkinter for the desktop graphical interface


## Project Structure

    fingerprint_iris_recognition/
        main.py                 Launches the graphical interface
        config.py               Central settings: image size, thresholds, paths
        requirements.txt        Python dependencies
        src/
            data/
                synth.py        Generates synthetic fingerprint and iris images
                dataset.py      Loads images and applies preprocessing
            models/
                cnn.py          Defines the convolutional neural network
            train.py            Trains a network for one modality
            enrollment.py       Builds the database of stored templates
            matcher.py          Matching, score fusion and the final decision
        gui/
            app.py              The CustomTkinter desktop application
            theme.py            Colours and fonts for the interface
        scripts/
            generate_data.py    Creates the synthetic dataset
            train_models.py     Trains both networks
            build_enrollment.py Builds the template database
            evaluate.py         Measures accuracy on unseen samples
            run_all.py          Runs every step end to end


## Installation and Setup

The project requires Python version 3.9 to 3.11.

Install the dependencies:

    pip install -r requirements.txt


## Running the Application

Step 1. Prepare the system. This generates the synthetic dataset, trains both
networks, and builds the enrollment database. It takes about one to two minutes
on a normal computer and only needs to be done once.

    python scripts/run_all.py

Step 2. Launch the interface.

    python main.py


## Using the Application

In the application window there are two upload panels, one for a fingerprint
and one for an iris. Select an image for each, then press Verify Identity. The
result panel shows the decision, a confidence value, and which enrolled subject
each sample matched.

To try the system immediately, press Load Demo Pair to load a ready made
example, then press Verify Identity. Example images for the other cases, such
as a mismatched pair and an unenrolled person, are available in the samples
folder.


## Results

On the generated dataset, both networks reach close to one hundred percent
accuracy when identifying enrolled subjects. In testing on captures the
networks had never seen:

- Genuine pairs, where the fingerprint and iris come from the same person, are
  correctly accepted as the same person.
- Cross pairs, where the fingerprint and iris come from two different enrolled
  people, are correctly reported as different people.
- An unenrolled person is never falsely accepted as a genuine match.

The decision thresholds were chosen by measuring the score distributions of
genuine and non genuine pairs rather than being set arbitrarily.


## Limitations

- The fingerprint and iris images are synthetic. The full recognition pipeline
  is genuine, but the inputs are generated, so this is a demonstration project
  rather than a deployable security product. To use real data, the dataset
  loader can be pointed at folders of real images organised by subject.
- Detecting that a person is not in the database at all is inherently harder
  than telling enrolled people apart, because the networks always map an input
  to the closest known identity. The system handles most such cases, but this is
  a known trade off in this type of recognition.


## Possible Improvements

- Replace the synthetic data with a real fingerprint and iris dataset.
- Add an enrollment screen so new people can be registered through the
  interface.
- Use a metric learning approach, such as triplet loss, to improve rejection of
  people who are not enrolled.
- Add liveness detection to guard against spoofed images.
