import pandas as pd


if __name__ == '__main__':
    train = pd.read_csv('../input/atmacup08-dataset/train.csv')
    test = pd.read_csv('../input/atmacup08-dataset/test.csv')
    print(train.shape, test.shape)
    # (8359, 16) (8360, 11)
    target_col = 'Global_Sales'

    train['lag_Year_of_Release_by_Publisher'] = train.groupby(['Name', 'Publisher'])['Year_of_Release'].diff(1)
    train['lead_Year_of_Release_by_Publisher'] = train.groupby(['Name', 'Publisher'])['Year_of_Release'].diff(-1)
    test['lag_Year_of_Release_by_Publisher'] = test.groupby(['Name', 'Publisher'])['Year_of_Release'].diff(1)
    test['lead_Year_of_Release_by_Publisher'] = test.groupby(['Name', 'Publisher'])['Year_of_Release'].diff(-1)

    use_cols = ['lag_Year_of_Release_by_Publisher', 'lead_Year_of_Release_by_Publisher']
    train_test = pd.concat([train, test], axis=0).reset_index(drop=True)
    train_test[use_cols].to_feather('../input/feather/diff.ftr')
