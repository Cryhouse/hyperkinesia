import pct.preprocessing.parsing as parsing
import pct.pipeline.pipeline as pipeline
import pct.pipeline.regressor as reg


def main():
    raw_data_path = "/Users/gustaf/Documents/skola/exjobb/tremor/data/raw_all"

    labels_all = parsing.parse_all_labels()
    filename_df = pipeline.get_filename_df(labels_all, raw_data_path, force=True)
    
    Y_full_class, Y_full_reg, Y_full_per, Y_full_jon = pipeline.get_Ys(labels_all)
    
    X_full = pipeline.get_X_full(filename_df, labels_all, Y_full_reg, force=True)

    reg.get_best_w(X_full, Y_full_reg)


if __name__ == '__main__':
    main()


    