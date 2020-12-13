import pandas as pd


if __name__ == '__main__':
    train = pd.read_csv('../input/atmacup08-dataset/train.csv')
    test = pd.read_csv('../input/atmacup08-dataset/test.csv')
    print(train.shape, test.shape)
    # (8359, 16) (8360, 11)
    target_col = 'Global_Sales'

    tr_gy_rank = train.sort_values(['Genre', 'Year_of_Release']).groupby(['Genre', 'Year_of_Release']).cumcount()
    tr_gy_cnt = train.sort_values(['Genre', 'Year_of_Release']).groupby(['Genre', 'Year_of_Release'])['Name'].transform('count')
    train['Name_serial_num_per'] = (tr_gy_rank / tr_gy_cnt).sort_index()

    te_gy_rank = test.sort_values(['Genre', 'Year_of_Release']).groupby(['Genre', 'Year_of_Release']).cumcount()
    te_gy_cnt = test.sort_values(['Genre', 'Year_of_Release']).groupby(['Genre', 'Year_of_Release'])['Name'].transform('count')
    test['Name_serial_num_per'] = (te_gy_rank / te_gy_cnt).sort_index()

    tr_ny_rank = train.sort_values(['Name', 'Year_of_Release']).groupby(['Name', 'Year_of_Release']).cumcount()
    tr_ny_cnt = train.sort_values(['Name', 'Year_of_Release']).groupby(['Name', 'Year_of_Release'])['Genre'].transform('count')
    train['Genre_serial_num_per'] = (tr_ny_rank / tr_ny_cnt).sort_index()

    te_ny_rank = test.sort_values(['Name', 'Year_of_Release']).groupby(['Name', 'Year_of_Release']).cumcount()
    te_ny_cnt = test.sort_values(['Name', 'Year_of_Release']).groupby(['Name', 'Year_of_Release'])['Genre'].transform('count')
    test['Genre_serial_num_per'] = (te_ny_rank / te_ny_cnt).sort_index()

    use_cols = ['Name_serial_num_per', 'Genre_serial_num_per']
    train_test = pd.concat([train, test], axis=0).reset_index(drop=True)
    train_test[use_cols].to_feather('../input/feather/rank.ftr')
