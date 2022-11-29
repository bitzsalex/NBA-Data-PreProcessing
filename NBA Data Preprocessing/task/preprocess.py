import pandas as pd
import numpy as np
import os
import requests
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Check for ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if 'nba2k-full.csv' not in os.listdir('../Data'):
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/nba2k-full.csv', 'wb').write(r.content)
    print('Loaded.')

data_path = "../Data/nba2k-full.csv"


# write your code here
def clean_data(path: str) -> pd.DataFrame:
    """
    Accepts a path to a dataset and returns the cleaned dataset

    :param path: path to the dataset
    :return: cleaned dataset
    """

    # We can parse the dates using parse_dates parameter
    # but in this case it didn't work
    df = pd.read_csv(path, parse_dates=True)

    # since the above parsing dates didn't work
    # let's do it one by one
    df.b_day = pd.to_datetime(df.b_day, format="%m/%d/%y")
    df.draft_year = pd.to_datetime(df.draft_year, format="%Y")

    # replacing the team missing values
    df.team.fillna("No Team", inplace=True)

    # Taking the proper measurements for height and weight features
    df.height = df.height.apply(lambda col: col.strip().split()[-1])
    df.weight = df.weight.apply(lambda col: col.strip().split()[-2])

    # Removing the extraneous $ symbol
    df.salary = df.salary.apply(lambda value: value.strip().replace("$", ""))

    # Changing the data types
    df[["height", "weight", "salary"]] = df[["height", "weight", "salary"]].astype(np.float64)

    # Categorizing the country feature
    df.country = df.country.apply(lambda value: value if value == "USA" else "Not-USA")

    # Replacing the "Undrafted" with "0"
    df.draft_round.replace("Undrafted", "0", inplace=True)

    return df


def feature_data(df: pd.DataFrame, cardinality=50) -> pd.DataFrame:
    # Extracting version year
    df.version = pd.to_datetime(
        df.version.apply(lambda value: value[-2:]),
        format="%y"
    )

    # engineering age column
    df["age"] = df.version.dt.year - df.b_day.dt.year

    # engineering experience
    df["experience"] = df.version.dt.year - df.draft_year.dt.year

    # engineering bmi
    df["bmi"] = df.weight / np.square(df.height)

    # dropping unnecessary columns
    df.drop(columns=["version", "b_day", "draft_year", "weight", "height"], inplace=True)

    # excluding columns from dropping in cardinality
    check_cardinal_cols = df.loc[:, ~df.columns.isin(["age", "experience", "bmi"])].columns

    # removing features with high cardinality
    # getting the features that will be dropped
    drop_features = [
        index for index, count in df[check_cardinal_cols].nunique().items() if count >= cardinality
    ]

    # remove target variable from the list drop_features df
    if 'salary' in drop_features:
        drop_features.remove('salary')

    # dropping the features, if the list isn't empty
    if len(drop_features):
        df.drop(columns=drop_features, inplace=True)

    return df


def multicol_data(df: pd.DataFrame, corr_coefficient: float = 0.5) -> pd.DataFrame:
    """
    Remove the multicollinearity of features in a data frame

    :param df: data frame to check the collinearity
    :param corr_coefficient: the minimum coefficient value to filter multicollinearity
    :return: a data frame that has non-multicollinearity
    """
    # filtering the numeric features
    numeric_features = list(df.select_dtypes('number').drop(columns='salary'))

    # calculating the correlation among each other
    corr = df[numeric_features].corr().abs().unstack().sort_values()

    # filter the correlation values
    corr = corr.where(corr > corr_coefficient).dropna()
    corr = corr.drop_duplicates()

    # get the name of features with high correlation value
    highest_features = list(set([
        feature
        for features in corr.index
        for feature in features
    ]))

    # calculate higher correlation features with target variable
    corr = df[[*highest_features, 'salary']].corr()

    # dropping the feature that has a minimum correlation with the target variable
    df.drop(columns=corr.loc[highest_features, 'salary'].idxmin(), inplace=True)

    return df


def transform_data(df: pd.DataFrame) -> tuple:
    """
    Transforming the numerical and categorical features using StandardScaler and OneHotEncoder respectively.

    :param df: a data frame to transform its values
    :return: a tuple of features and target variable
    """

    # getting the numerical and categorical variables
    numerical_features = df.select_dtypes("number").drop(columns="salary")
    ordinal_features = df.select_dtypes("object")

    # performing the StandardScaler for numerical features
    std_scaler = StandardScaler()
    std_scaler.fit(numerical_features)
    numerical_features[numerical_features.columns] = std_scaler.transform(numerical_features)

    # performing the OneHotEncoding for categorical features
    oh_encoder = OneHotEncoder(sparse=False)
    features = ordinal_features.columns.copy()

    # looping the features
    for feature in features:
        # reshape the features
        trf_feature = ordinal_features[feature].values.reshape(-1, 1)

        # fit the feature and transform it
        oh_encoder.fit(trf_feature)
        transformed = oh_encoder.transform(trf_feature)

        # drop the feature from the previous data frame
        ordinal_features.drop(columns=feature, inplace=True)

        # create new data frame and concatenate with the previous one
        ordinal_features = pd.concat(
            [ordinal_features, pd.DataFrame(transformed, columns=oh_encoder.categories_[0])], axis=1)

    # return the features and target variables
    return pd.concat([numerical_features, ordinal_features], axis=1), df.salary
