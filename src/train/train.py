import sys
import pandas as pd
from datetime import datetime
from helpers import get_holidays_count


CTB_PARAMS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 1200,
    "random_seed": 42,
    "od_wait": 50,
    "od_type": "Iter",
    "thread_count": 10,
    "learning_rate": 0.06,
    "depth": 12,
}


class CtbTrain:
    X = None
    y = None
    X_test = None
    y_test = None
    ctb_data = None
    model = None
    rmse = None
    SPLIT_TRAIN_TEST_ON = '2019-12-01'  # month date where to split train/test data

    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename

    def load_file(self):
        sys.stdout.write("File {} loading...".format(self.filename))
        self.df = pd.read_csv(self.filename)

    def init_features(self):
        sys.stdout.write("Features initializing...")
        self.load_file()
        df = self.df
        df['date'] = df.apply(lambda x: datetime(int(x['year']), int(x['month']), 1), axis=1)
        
        # find empty uninformative brand with no sales for whole period
        df_brand_sum = df.groupby(['brand'])['sales'].sum().reset_index()
        useless_brands = df_brand_sum[df_brand_sum['sales'] <= 0]['brand'].tolist()
        df = df.drop(df[df['brand'] == useless_brands[0]].index)

        # drop Nan rows
        df = df.dropna(subset=['sales', 'district'])

        # lag features for store sales
        month_revenues = df.groupby(['address', 'date']).agg({'sales': ['sum', 'mean', 'std']}).reset_index()
        month_revenues.columns = ['address', 'date', 'sales_sum', 'sales_mean', 'sales_std']

        month_revenues['lag_1_store_sales_sum'] =  month_revenues.groupby('address')['sales_sum'].shift(1)
        month_revenues['lag_1_store_sales_mean'] =  month_revenues.groupby('address')['sales_mean'].shift(1)
        month_revenues['lag_1_store_sales_std'] =  month_revenues.groupby('address')['sales_std'].shift(1)

        month_revenues['lag_2_store_sales_sum'] =  month_revenues.groupby('address')['sales_sum'].shift(2)
        month_revenues['lag_2_store_sales_mean'] =  month_revenues.groupby('address')['sales_mean'].shift(2)
        month_revenues['lag_2_store_sales_std'] =  month_revenues.groupby('address')['sales_std'].shift(2)

        month_revenues['lag_3_store_sales_sum'] =  month_revenues.groupby('address')['sales_sum'].shift(3)
        month_revenues['lag_3_store_sales_mean'] =  month_revenues.groupby('address')['sales_mean'].shift(3)
        month_revenues['lag_3_store_sales_std'] =  month_revenues.groupby('address')['sales_std'].shift(3)

        left_df =  df.set_index(['address', 'date'])
        right_df = month_revenues.set_index(['address', 'date'])
        merged_df = pd.merge(left_df, right_df, left_index=True, right_index=True, how='left', sort=False).reset_index()

        # lag features for brand sales in each store
        df_grouped_by_brand = df.groupby(['address', 'brand', 'date']).agg({'sales':'sum'}).reset_index()
        
        df_grouped_by_brand['lag_1_brand_sales'] = df_grouped_by_brand.groupby(['address', 'brand'])['sales'].shift(1)
        df_grouped_by_brand['lag_2_brand_sales'] = df_grouped_by_brand.groupby(['address', 'brand'])['sales'].shift(2)
        df_grouped_by_brand['lag_3_brand_sales'] = df_grouped_by_brand.groupby(['address', 'brand'])['sales'].shift(3)

        left_df =  merged_df.set_index(['address', 'brand', 'date'])
        right_df = df_grouped_by_brand.drop('sales', axis=1).set_index(['address', 'brand', 'date'])
        df = pd.merge(left_df, right_df, left_index=True, right_index=True, how='left', sort=False).reset_index()

        # days, holidays count
        days = pd.Series([0, 31,28,31,30,31,30,31,31,30,31,30,31])
        df['days'] = df['month'].map(days).astype(np.int64)
        df['days'] = df['days'].astype(np.int64)

        df['holidays_cnt'] = df['date'].apply(get_holidays_count).astype(np.int64)

        self.df = df.drop(['location', 'Unnamed: 0'], axis=1)

    def init_train_val_data(self):
        sys.stdout.write("Data splitting preparation...")
        train_set = df[df['date'] < self.SPLIT_TRAIN_TEST_ON].reset_index()
        test_set = df[df['date'] >= self.SPLIT_TRAIN_TEST_ON].reset_index()
        assert len(test_set) + len(train_set) == len(df), True

        self.X = train_set.drop(['sales', 'date', 'address'], axis=1)
        self.y = train_set['sales']
        self.X_test = test_set.drop(['sales', 'date', 'address'], axis=1)
        self.y_test = test_set['sales']
        assert len(X_test) + len(X) == len(df), True
        assert len(y) + len(y_test) == len(df), True

    def init_ctb_model_and_data(self):
        sys.stdout.write("Model and data preparation...")
        assert X, True
        assert y, True
        categorical_features_indices = np.where((X.dtypes != np.float) & (X.dtypes != np.int))[0]
        self.ctb_data = ctb.Pool(self.X, self.y, cat_features=categorical_features_indices)
        self.model = ctb.CatBoostRegressor(**CTB_PARAMS)

    def run_train_and_test(self):
        sys.stdout.write("Starting training...")
        self.init_features()
        self.init_train_val_data()
        self.init_ctb_model_and_data()
        assert self.model, True
        self.model.fit(self.ctb_data)
        y_pred = model.predict(self.X_test)
        rmse = get_rmse(self.y_test, y_pred)
        return "Training finished\nTest RMSE: {}\n".format(rmse)


if __name__ == '__main__':
    filename = sys.argv[1]
    sys.stdout.write("filename from stdout")
    ctb_train = CtbTrain(filename)
    res = ctb_train.run_train_and_test()
    sys.stdout.write(res)
