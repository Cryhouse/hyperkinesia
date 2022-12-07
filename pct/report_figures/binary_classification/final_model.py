import sklearn.svm as svm

import pct.pipeline.feature_ext as f
import pct.preprocessing.parsing as parsing
import pct.pipeline.pipeline as pipeline
import pct.pipeline.binary_classifier as bc



def get_final_model(test=False):
    raw_data_path = "/Users/gustaf/Documents/skola/exjobb/tremor/data/raw_all"

    labels = parsing.parse_labels_minus_test() if test else parsing.parse_all_labels()
    t = "train_val" if test else "all"
    filename_df = pipeline.get_filename_df(labels, raw_data_path, force=False, t=t)
    
    Y_full_class, Y_full_reg, Y_full_per, Y_full_jon = pipeline.get_Ys(labels)
    X_train = pipeline.get_X_full(filename_df, labels, Y_full_class, force=True, t="regressor")
    X_train_norm, m, s = f.normalize_features(X_train)

    C = 2

    model = svm.SVC(C=C, kernel="rbf")
    model.fit(X_train_norm, Y_full_class)
    return model, m, s


def get_final_model2(train_inds, val_inds):
    raw_data_path = "/Users/gustaf/Documents/skola/exjobb/tremor/data/raw_all"

    labels = parsing.parse_all_labels()

    labels_train = labels.iloc[train_inds,:].reset_index(drop=True)
    labels_val = labels.iloc[val_inds,:].reset_index(drop=True)
    filename_df = pipeline.get_filename_df(labels, raw_data_path, force=False)
    
    Y_train_class, Y_train_reg, Y_train_per, Y_train_jon = pipeline.get_Ys(labels_train)
    X_train = pipeline.get_X_full(filename_df, labels_train, Y_train_reg, force=True, t="regressor")
    X_train_norm, m, s = f.normalize_features(X_train)

    C = 2

    model = svm.SVC(C=C, kernel="rbf", probability=True)
    model.fit(X_train_norm, Y_train_class)
    return model
   

def main():
    pass
    
  

    
    
    


if __name__ == '__main__':
    main()


    