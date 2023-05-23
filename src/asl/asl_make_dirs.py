import os
import shutil

TESTDATA_PATH = os.path.join('asl_alphabet_test')

DIRS = ['A','B','C','D','F','G','H','I','J','K','L','M','N','O','P','Q',
        'R','S','T','U','V','W','X','Y','Z','none','space']

# Create dirs for test data and moves images to their respective dirs
def main():
    for dir in DIRS:
        os.mkdir(os.path.join(TESTDATA_PATH, dir))
        current_destination = os.path.join(TESTDATA_PATH, f'{dir}_test.jpg')
        new_destination = os.path.join(TESTDATA_PATH, dir, f'{dir}_test.jpg')
        shutil.move(current_destination, new_destination)

if __name__ == '__main__':
    # main()
    print('No longer usefull.')