#!/usr/bin/env python
# coding: utf-8
import os
import pickle
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from app.settings import MEDIA_ROOT

warnings.filterwarnings("ignore")

is_data_loaded = False

global ax, fig, x_train, x_test, y_train, y_test, dataset, output_dir


def load_dataset():
    global ax, fig, x_train, x_test, y_train, y_test, dataset, output_dir
    if not is_data_loaded:
        print('Loading Dataset ...')
        dataset = pd.read_csv(MEDIA_ROOT + 'dataset.csv', parse_dates=['Dates'])
        output_dir = MEDIA_ROOT + 'output_dir/'
        try:
            os.makedirs(output_dir)
        except OSError as e:
            pass

        print('Loaded Dataset!')
    else:
        print('Loaded Already Dataset!')


def get_metrics(y_true, y_pred, algo_name=''):
    return {'mae': mean_absolute_error(y_true, y_pred), 'evs': explained_variance_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred), 'r2score': r2_score(y_true, y_pred), 'algo_name': algo_name}


def merge_results_val(results, x_test, values, algo_name):
    temp_df = x_test.copy()
    temp_df['Algorithm'] = algo_name
    temp_df['Values'] = values
    results = results.append(temp_df, ignore_index=True)
    results = results[['Algorithm', 'Year', 'Values']]
    return results


def perform_preprocessing():
    global ax, fig, x_train, x_test, y_train, y_test, dataset
    load_dataset()
    print('Performing Data Preprocessing ...')
    dataset['Year'] = dataset['Dates'].map(lambda x: x.year)
    dataset['Week'] = dataset['Dates'].map(lambda x: x.week)
    dataset['Hour'] = dataset['Dates'].map(lambda x: x.hour)
    # number of cases every 2 weeks
    dataset['event'] = 1
    weekly_events = dataset[['Week', 'Year', 'event']].groupby(['Year', 'Week']).count().reset_index()
    weekly_events_years = weekly_events.pivot(index='Week', columns='Year', values='event').fillna(method='ffill')
    weekly_events_years.head()
    ax = weekly_events_years.interpolate().plot(title='number of cases every 2 weeks', figsize=(10, 6))
    plt.savefig(output_dir + 'events_every_two_weeks.png')
    # hourly_events
    hourly_events = dataset[['Hour', 'event']].groupby(['Hour']).count().reset_index()
    hourly_events.plot(kind='bar', figsize=(12, 6))
    plt.savefig(output_dir + 'hourly_events.png')
    # hourly_district_events_pivot
    hourly_district_events = dataset[['PdDistrict', 'Hour', 'event']].groupby(
        ['PdDistrict', 'Hour']).count().reset_index()
    hourly_district_events_pivot = hourly_district_events.pivot(index='Hour', columns='PdDistrict',
                                                                values='event').fillna(method='ffill')
    hourly_district_events_pivot.interpolate().plot(title='number of cases hourly by district', figsize=(12, 6))
    plt.savefig(output_dir + 'hourly_events_by_district.png')
    # Crime Categories
    categories = dataset["Category"].unique()
    # how many times each crime happened
    crimes_counts = dataset['Category'].value_counts()
    # Visualize crime rate by category
    data = crimes_counts
    plt.figure(figsize=(10, 10))
    with sns.axes_style("whitegrid"):
        ax = sns.barplot(
            y=data.index,
            x=(data.values / data.values.sum()) * 100,
            orient="h",
            palette=sns.color_palette("Blues"))
    plt.title("Bangladesh Crime Rate by Category", fontdict={"fontsize": 16})
    plt.xlabel("Incidents (%)")
    plt.savefig(output_dir + 'Bangladesh_Crime_Rate_by_Category.png')
    # Visualize crime rate by hour
    dataset["Date"] = dataset.Dates.dt.date
    dataset["Hour"] = dataset.Dates.dt.hour
    data = dataset.groupby(["Hour", "Date", "Category"], as_index=False).count().iloc[:, :4]
    data.rename(columns={"Dates": "Incidents"}, inplace=True)
    data = data.groupby(["Hour", "Category"], as_index=False).mean()
    # data = data.loc[]
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(30, 20))
    ax = sns.lineplot(x="Hour", y="Incidents", data=data, hue="Category")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=6)
    plt.title("Bangladesh Crime Rate by Hour of Day (Average Incidents Per Hour)", fontdict={"fontsize": 16})
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_dir + 'Bangladesh_Crime_Rate_by_Hour.png')
    yearly_events = dataset[['Year', 'Category', 'event']].groupby(['Year', 'Category']).count().reset_index()
    yearly_events.head()
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(30, 20))
    ax = sns.lineplot(x="Year", y="event", data=yearly_events, hue="Category")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=6)
    plt.title("Bangladesh Crime Rate by Year (Average Incidents Per Year)", fontdict={"fontsize": 24})
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.xticks(yearly_events['Year'].unique())
    plt.savefig(output_dir + 'Bangladesh_Crime_Rate_by_Year.png')
    X = yearly_events[['Year', 'Category']]
    Y = yearly_events['event']
    category_le = preprocessing.LabelEncoder()
    category_le.fit(X['Category'])
    X['Category'] = category_le.transform(X['Category'])

    with open(MEDIA_ROOT + 'category_le.pkl', 'wb') as f:
        pickle.dump(category_le, f)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
    print(len(x_train), len(x_test), len(y_train), len(y_test))

    print('Data Preprocessing Complete!')


def generate_graph(data_df, col_name='mae', y_label_name='', x_label_name=''):
    cmap = plt.get_cmap("Dark2")
    norm = plt.Normalize(data_df[col_name].min(), data_df[col_name].max())
    g = sns.catplot(x='algo_name', y=col_name, data=data_df, kind='bar', legend=True, height=5, aspect=.8,
                    palette=cmap(norm(data_df[col_name].values)))

    g.ax.xaxis.label.set_color('red')
    g.ax.yaxis.label.set_color('red')
    g.ax.set_xlabel(x_label_name)
    g.ax.set_ylabel(y_label_name)
    g.ax.xaxis.get_label().set_fontfamily("Times New Roman")
    g.ax.xaxis.get_label().set_fontsize(14)

    g.ax.yaxis.get_label().set_fontfamily("Times New Roman")
    g.ax.yaxis.get_label().set_fontsize(14)
    g.fig.set_size_inches(20, 8)
    g.fig.subplots_adjust(top=0.81, right=0.86)

    plt.savefig(output_dir + 'Metrics_{}.png'.format(y_label_name))


def perform_regression_analysis():
    global ax, fig, x_train, x_test, y_train, y_test
    try:
        perform_preprocessing()
        print('Performing Linear Regression ...')
        # LinearRegression
        linear_regressor = linear_model.LinearRegression()
        linear_regressor.fit(x_train, y_train)
        linear_regressor_pred = linear_regressor.predict(x_test)
        linear_metrics = get_metrics(y_test, linear_regressor_pred, 'Linear Regression')
        print('Linear Regression Complete!')

        print('Performing Polynomial Regression ...')
        # Polynomial Regression
        transformer = preprocessing.PolynomialFeatures(degree=2, include_bias=False)
        x_train_ = transformer.fit_transform(x_train)
        x_test_ = transformer.fit_transform(x_test)

        polynomial_regressor = linear_model.LinearRegression()
        polynomial_regressor.fit(x_train_, y_train)
        polynomial_regressor_pred = polynomial_regressor.predict(x_test_)
        polynomial_regressor_linear_metrics = get_metrics(y_test, polynomial_regressor_pred, 'Polynomial Regression')
        print('Polynomial Regression Complete!')

        print('Performing RandomForest Regression ...')
        # RandomForestRegressor
        random_forest_regressor = RandomForestRegressor(max_depth=2, random_state=0)
        random_forest_regressor.fit(x_train, y_train)
        random_forest_regressor_pred = random_forest_regressor.predict(x_test)
        random_forest_regressor_metrics = get_metrics(y_test, random_forest_regressor_pred, 'Random Forest Regression')
        print('Random Forest Regression Complete!')

        results = pd.DataFrame(columns=['Algorithm', 'Year', 'Values'])

        print('Generating Metrics ...')
        results = merge_results_val(results, x_test, y_test, 'Actual')
        results = merge_results_val(results, x_test, linear_regressor_pred, 'Linear Regression')
        results = merge_results_val(results, x_test, polynomial_regressor_pred, 'Polynomial Regression')
        results = merge_results_val(results, x_test, random_forest_regressor_pred, 'Random Forest Regression')
        results['Values'] = results['Values'].astype('int')

        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(30, 20))
        ax = sns.lineplot(x="Year", y="Values", data=results, hue="Algorithm")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=6)
        plt.title("Algorithm Prediction Comparison", fontdict={"fontsize": 24})
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_dir + 'Algorithm Prediction Comparison.png')

        metrics = pd.DataFrame()
        metrics = metrics.append(linear_metrics, ignore_index=True)
        metrics = metrics.append(polynomial_regressor_linear_metrics, ignore_index=True)
        metrics = metrics.append(random_forest_regressor_metrics, ignore_index=True)

        generate_graph(metrics, col_name='mae', y_label_name='MAE Score', x_label_name='Algorithm Name')
        generate_graph(metrics, col_name='mse', y_label_name='MSE Score', x_label_name='Algorithm Name')
        generate_graph(metrics, col_name='evs', y_label_name='EVS Score', x_label_name='Algorithm Name')
        generate_graph(metrics, col_name='r2score', y_label_name='R2 Score', x_label_name='Algorithm Name')

        with open(MEDIA_ROOT + 'polynomial_features.pkl', 'wb') as f:
            pickle.dump(transformer, f)

        with open(MEDIA_ROOT + 'crime_random_forest.pkl', 'wb') as f:
            pickle.dump(random_forest_regressor, f)

        with open(MEDIA_ROOT + 'linear_regressor.pkl', 'wb') as f:
            pickle.dump(linear_regressor, f)

        with open(MEDIA_ROOT + 'polynomial_regressor.pkl', 'wb') as f:
            pickle.dump(polynomial_regressor, f)

        print('Metrics Generation Complete!')
        print('Graph output is available in {} directory '.format(output_dir))
    except Exception as e:
        print('Error while Training : ', e)
        return False
    return True


def predict_crime_rate_by_year(year, category, algorithm):
    crime_rate = 0
    with open(MEDIA_ROOT + 'category_le.pkl', 'rb') as f:
        category_le = pickle.load(f)

    with open(MEDIA_ROOT + 'crime_random_forest.pkl', 'rb') as f:
        random_forest_regressor = pickle.load(f)

    with open(MEDIA_ROOT + 'linear_regressor.pkl', 'rb') as f:
        linear_regressor = pickle.load(f)

    with open(MEDIA_ROOT + 'polynomial_regressor.pkl', 'rb') as f:
        polynomial_regressor = pickle.load(f)

    with open(MEDIA_ROOT + 'polynomial_features.pkl', 'rb') as f:
        polynomial_features = pickle.load(f)

    category_encoded = category_le.transform([category])
    print('category_encoded : ', category_encoded)

    if 'linear' in algorithm:
        if linear_regressor is not None:
            crime_rate = linear_regressor.predict([[year, category_encoded]])
    elif 'polynomial' in algorithm:
        if polynomial_regressor is not None:
            crime_rate = polynomial_regressor.predict(polynomial_features.transform([[year, category_encoded]]))
    elif 'random_forest' in algorithm:
        if random_forest_regressor is not None:
            crime_rate = random_forest_regressor.predict([[year, category_encoded]])

    print('crime_rate : ', crime_rate)
    return int(crime_rate)
