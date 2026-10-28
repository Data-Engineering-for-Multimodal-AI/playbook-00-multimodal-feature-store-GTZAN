# sample_data

`python download_data.py` puts GTZAN here (1.2 GB; not part of the repository):

- `genres_original/<genre>/<genre>.000NN.wav`: 1,000 30-second clips, 100 per genre (blues, classical,
  country, disco, hiphop, jazz, metal, pop, reggae, rock). `jazz.00054.wav` is unreadable.
- `images_original/<genre>/<genre>000NN.png`: the mel spectrogram of each clip (999), from the Kaggle
  version of GTZAN. They are images of the audio, not album art.

GTZAN: G. Tzanetakis and P. Cook, "Musical genre classification of audio signals", IEEE Transactions on
Speech and Audio Processing, 2002. The clips come from commercial recordings; GTZAN is used for research
and isn't redistributed here. Kaggle: andradaolteanu/gtzan-dataset-music-genre-classification.
