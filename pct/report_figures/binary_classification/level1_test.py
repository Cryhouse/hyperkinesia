import sklearn.svm as svm

import pct.pipeline.feature_ext as f
import pct.preprocessing.parsing as parsing
import pct.pipeline.pipeline as pipeline

# predefined test set
from pct.eval.pretty_table import print_pretty_table
from pct.eval.eval_binary import eval_binary

def main():
    raw_data_path = "/Users/gustaf/Documents/skola/exjobb/tremor/data/raw_all"

    labels_train = parsing.parse_labels_minus_test()
    Y_full_class, Y_full_reg, Y_full_per, Y_full_jon = pipeline.get_Ys(labels_train)
    filename_df = pipeline.get_filename_df_trunc(labels_train, raw_data_path, force=True)
    X_full = pipeline.get_X_full(filename_df, labels_train, Y_full_reg, force=True)
    X_full_norm, m, s = f.normalize_features(X_full)

    labels_test = parsing.parse_test_labels()
    Y_test_class, Y_test_reg, Y_test_per, Y_test_jon = pipeline.get_Ys(labels_test)
    filename_df_test = pipeline.get_filename_df_trunc(labels_test, raw_data_path, force=True)
    X_test = pipeline.get_X_full(filename_df_test, labels_test, Y_test_reg, force=True)
    C = 2
    model = svm.SVC(C=2,kernel="rbf")
    model.fit(X_full_norm, Y_full_class)

    X_test_norm, _, _ = f.normalize_features(X_test, m ,s)
    pred = model.predict(X_test_norm)
    description, e = eval_binary(pred, Y_test_class, Y_test_reg)
    
    
    
    print_pretty_table(e, description)
    
    
    


if __name__ == '__main__':
    main()


    