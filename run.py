import sys


def train_word2vec(num_features, min_word_count, num_workers, context, downsampling):
    from preprocessing import train_word2vec
    import pandas as pd
    df = pd.read_csv("datafile.csv", header=0, delimiter="\t", quoting=3)
    model = train_word2vec(df, num_features, min_word_count, num_workers, context, downsampling)
    return model

def train_model():
    import model


if __name__ == "__main__":
    x = 0
    if sys.argv[1] == "word2vec":
        train_word2vec(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]))
    elif sys.argv[1] == "model":
        train_model()
    else:
        x = 0


