import os


LABELS = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "ñ",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "amor",
    "hola",
    "yo",
    "none",
    "espacio",
]
DATA_PATH = "lsm_train"


def main():
    for label in LABELS:
        label_directory = os.path.join(DATA_PATH, label)
        os.makedirs(label_directory, exist_ok=True)


if __name__ == "__main__":
    main()
