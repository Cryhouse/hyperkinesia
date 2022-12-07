import sklearn.svm as svm

import pct.pipeline.feature_ext as f
import pct.preprocessing.parsing as parsing
import pct.pipeline.pipeline as pipeline
from pct.eval.eval_binary import eval_binary

from pct.config import RAW_DATA_DIR, LABELS_PATH, TRAIN_PATIENTS, TEST_PATIENTS
def main():
    labels_minus_test = parsing.parse_labels_minus_test()
    labels_test = parsing.parse_test_labels()

    filename_df_minus_test = pipeline.get_filename_df(labels_minus_test, RAW_DATA_DIR, force=False)
    
    Y_full_class, Y_full_reg, Y_full_per, Y_full_jon = pipeline.get_Ys(labels_minus_test)
    X_train = pipeline.get_X_full(filename_df_minus_test, labels_minus_test, Y_full_class, force=True)
    X_train_norm, m, s = f.normalize_features(X_train)

    C = 2

    model = svm.SVC(C=C, kernel="rbf")
    model.fit(X_train_norm, Y_full_class)
   
    labels_test = parsing.parse_test_labels()
    
    filename_df_test = pipeline.get_filename_df(labels_test, RAW_DATA_DIR, force=True)
    Y_test_class, Y_test_reg, Y_test_per, Y_test_jon = pipeline.get_Ys(labels_test)
    print(f"test data patients: {labels_test['patient'].unique()}")
    print(f"number of test data samples: {len(Y_test_reg)}")
    X_test = pipeline.get_X_full(filename_df_test, labels_test, Y_test_class, force=True)
    X_test_norm, _, _ = f.normalize_features(X_test, m, s)

    test_pred = model.predict(X_test_norm)
    description, eval = eval_binary(test_pred, Y_test_class, Y_test_reg, plot=True) # the regression values are needed to see the misclassification distribution
    print(f"description: {description}")
    print(f"eval: {eval}")

    
    
    


if __name__ == '__main__':
    main()


    