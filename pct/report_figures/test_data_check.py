import pct.preprocessing.parsing as parsing
from pct.config import RAW_DATA_DIR, LABELS_PATH, TRAIN_PATIENTS, TEST_PATIENTS

def main():
    labels_test = parsing.parse_test_labels()
    test_patients = labels_test["patient"].unique()
    print(len(labels_test))
    labels_test.to_csv("test_patients.csv",sep=";")
    print(test_patients)
if __name__ == '__main__':
    main()

