from datetime import datetime
import json
import os
import requests

import numpy as np
import pandas as pd

from . import constants

FILEPATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(FILEPATH, '../data/'))


def get_all_data():
    '''
    Main routine that grabs all COVID and covariate data and
    returns them as a single dataframe that contains:

    * count of cumulative cases and deaths by country (by today's date)
    * days since first case for each country
    * CPI gov't transparency index
    * World Bank data on population, healthcare, etc. by country
    '''

    all_covid_data = _get_latest_covid_timeseries()

    covid_cases_rollup = _rollup_by_country(all_covid_data['Confirmed'])
    covid_deaths_rollup = _rollup_by_country(all_covid_data['Deaths'])

    todays_date = covid_cases_rollup.columns.max()

    # Create DataFrame with today's cumulative case and death count, by country
    df_out = pd.DataFrame({'cases': covid_cases_rollup[todays_date],
                           'deaths': covid_deaths_rollup[todays_date]})

    _clean_country_list(df_out)
    _clean_country_list(covid_cases_rollup)

    # Add observed death rate:
    df_out['death_rate_observed'] = df_out.apply(
        lambda row: row['deaths'] / float(row['cases']),
        axis=1)

    # Add covariate for days since first case
    df_out['days_since_first_case'] = _compute_days_since_nth_case(
        covid_cases_rollup, n=1)

    # Add CPI covariate:
    _add_cpi_data(df_out)

    # Add World Bank covariates:
    _add_wb_data(df_out)

    # Drop any country w/o covariate data:
    num_null = df_out.isnull().sum(axis=1)
    to_drop_idx = df_out.index[num_null > 1]
    print('Dropping %i/%i countries due to lack of data' %
          (len(to_drop_idx), len(df_out)))
    df_out.drop(to_drop_idx, axis=0, inplace=True)

    return df_out


def get_data_case_count_model():
    ''' Get data for case count model '''

    all_covid_data = _get_latest_covid_timeseries()

    covid_cases_rollup = _rollup_by_country(all_covid_data['Confirmed'])
    covid_deaths_rollup = _rollup_by_country(all_covid_data['Deaths'])

    _clean_country_list(covid_cases_rollup)
    _clean_country_list(covid_deaths_rollup)

    todays_date = covid_cases_rollup.columns.max()

    # Create DataFrame with today's cumulative case and death count, by country
    df_out = pd.DataFrame({'cases': covid_cases_rollup[todays_date],
                           'deaths': covid_deaths_rollup[todays_date]})

    # Add observed death rate:
    df_out['death_rate_observed'] = df_out.apply(
        lambda row: row['deaths'] / float(row['cases']),
        axis=1)

    # Add covariate for days since first case
    df_out['days_since_hundredth_case'] = _compute_days_since_nth_case(
        covid_cases_rollup, n=100)

    # Add Testing metric:
    _add_testing_data(df_out)

    df_out = df_out.loc[df_out['tests_per_million'].notnull()]

    return(df_out)


def get_statewise_testing_data():
    ''' Pull all statewise data required for model fitting and
    prediction

    Returns:
    * df_out: DataFrame for model fitting where inclusion
        requires testing data from 7 days ago
    * df_pred: DataFrame for count prediction where inclusion
        only requires testing data from today
    '''

    # Pull testing counts by state:
    out = requests.get('https://covidtracking.com/api/states')
    df_out = pd.DataFrame(out.json())
    df_out.set_index('state', drop=True, inplace=True)

    # Pull time-series of testing counts:
    ts = requests.get('https://covidtracking.com/api/states/daily')
    df_ts = pd.DataFrame(ts.json())

    # Get data from last week
    date_last_week = df_ts['date'].unique()[7]
    df_ts_last_week = _get_test_counts(df_ts, df_out.index, date_last_week)
    df_out['num_tests_7_days_ago'] = \
        (df_ts_last_week['positive'] + df_ts_last_week['negative'])
    df_out['num_pos_7_days_ago'] = df_ts_last_week['positive']

    # Get data from today
    date_today = df_ts['date'].unique()[1]
    df_ts_today = _get_test_counts(df_ts, df_out.index, date_today)
    df_out['num_tests_today'] = \
        (df_ts_today['positive'] + df_ts_today['negative'])

    # State population:
    df_pop = pd.read_excel('data/us_population_by_state_2019.xlsx',
                           skiprows=2, skipfooter=5)
    with open('data/us-state-name-abbr.json', 'r') as f:
        state_name_abbr_lookup = json.load(f)

    df_pop.index = df_pop['Geographic Area'].apply(
        lambda x: str(x).replace('.', '')).map(state_name_abbr_lookup)
    df_pop = df_pop.loc[df_pop.index.dropna()]

    df_out['total_population'] = df_pop['Total Resident\nPopulation']

    # Tests per million people, based on today's test coverage
    df_out['tests_per_million'] = 1e6 * \
        (df_out['num_tests_today']) / df_out['total_population']
    df_out['tests_per_million_7_days_ago'] = 1e6 * \
        (df_out['num_tests_7_days_ago']) / df_out['total_population']

    # People per test:
    df_out['people_per_test'] = 1e6 / df_out['tests_per_million']
    df_out['people_per_test_7_days_ago'] = \
        1e6 / df_out['tests_per_million_7_days_ago']

    # Drop states with messed up / missing data:
    # Drop states with missing total pop:
    to_drop_idx = df_out.index[df_out['total_population'].isnull()]
    print('Dropping %i/%i states due to lack of population data: %s' %
          (len(to_drop_idx), len(df_out), ', '.join(to_drop_idx)))
    df_out.drop(to_drop_idx, axis=0, inplace=True)

    df_pred = df_out.copy(deep=True)  # Prediction DataFrame

    # Criteria for model fitting:
    # Drop states with missing test count 7 days ago:
    to_drop_idx = df_out.index[df_out['num_tests_7_days_ago'].isnull()]
    print('Dropping %i/%i states due to lack of tests: %s' %
          (len(to_drop_idx), len(df_out), ', '.join(to_drop_idx)))
    df_out.drop(to_drop_idx, axis=0, inplace=True)
    # Drop states with no cases 7 days ago:
    to_drop_idx = df_out.index[df_out['num_pos_7_days_ago'] == 0]
    print('Dropping %i/%i states due to lack of positive tests: %s' %
          (len(to_drop_idx), len(df_out), ', '.join(to_drop_idx)))
    df_out.drop(to_drop_idx, axis=0, inplace=True)

    # Criteria for model prediction:
    # Drop states with missing test count today:
    to_drop_idx = df_pred.index[df_pred['num_tests_today'].isnull()]
    print('Dropping %i/%i states due to lack of tests: %s' %
          (len(to_drop_idx), len(df_pred), ', '.join(to_drop_idx)))
    df_pred.drop(to_drop_idx, axis=0, inplace=True)

    return df_out, df_pred


def get_county_data():
    ''' Get data for COVID-19 at the U.S. county level '''
    df_covid = pd.read_csv(
        ('https://raw.githubusercontent.com/nytimes/'
         'covid-19-data/master/us-counties.csv'))
    df_covid.dropna(axis=0, subset=['fips'], inplace=True)
    latest_date = df_covid['date'].max()
    df_covid['fips'] = df_covid['fips'].astype(int)

    print('Getting data for %s' % latest_date)

    df_covid_today = df_covid.loc[df_covid['date'] == latest_date]

    df_population = pd.read_csv(
        os.path.join(DATA_DIR, 'co-est2019-alldata.csv'),
        usecols=['COUNTY', 'STATE', 'POPESTIMATE2019'])
    df_population['fips'] = list(
        map(lambda x, y: 1000 * x + y,
            df_population['STATE'], df_population['COUNTY']))

    df = df_covid_today.merge(df_population, how='left', on='fips')

    df['Cases per 100k'] = list(
        map(lambda x, y: round(x * 1e5 / y, 1),
            df['cases'], df['POPESTIMATE2019']))

    df['Deaths per 100k'] = list(
        map(lambda x, y: round(x * 1e5 / y, 1),
            df['deaths'], df['POPESTIMATE2019']))

    df['fips'] = df['fips'].apply(
        lambda x: str(x) if x >= 10000 else '0%i' % x)

    return df


def _get_test_counts(df_ts, state_list, date):

    ts_list = []
    for state in state_list:
        state_ts = df_ts.loc[df_ts['state'] == state]
        # Back-fill any gaps to avoid crap data gaps
        state_ts.fillna(method='bfill', inplace=True)

        record = state_ts.loc[df_ts['date'] == date]
        ts_list.append(record)

    df_ts = pd.concat(ts_list, ignore_index=True)
    return df_ts.set_index('state', drop=True)


def _get_latest_covid_timeseries():
    ''' Pull latest time-series data from JHU CSSE database '''

    repo = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/'
    data_path = 'csse_covid_19_data/csse_covid_19_time_series/'

    all_data = {}
    for status in ['Confirmed', 'Deaths', 'Recovered']:
        file_name = 'time_series_19-covid-%s.csv' % status
        all_data[status] = pd.read_csv(
            '%s%s%s' % (repo, data_path, file_name))

    return all_data


def _rollup_by_country(df):
    '''
    Roll up each raw time-series by country, adding up the cases
    across the individual states/provinces within the country

    :param df: Pandas DataFrame of raw data from CSSE
    :return: DataFrame of country counts
    '''
    gb = df.groupby('Country/Region')
    df_rollup = gb.sum()
    df_rollup.drop(['Lat', 'Long'], axis=1, inplace=True, errors='ignore')

    return _convert_cols_to_dt(df_rollup)


def _convert_cols_to_dt(df):

    # Convert column strings to dates:
    idx_as_dt = [datetime.strptime(x, '%m/%d/%y') for x in df.columns]
    df.columns = idx_as_dt
    return df


def _clean_country_list(df):
    ''' Clean up input country list in df '''
    # handle recent changes in country names:
    country_rename = {
        'Hong Kong SAR': 'Hong Kong',
        'Taiwan*': 'Taiwan',
        'Czechia': 'Czech Republic',
        'Brunei': 'Brunei Darussalam',
        'Iran (Islamic Republic of)': 'Iran',
        'Viet Nam': 'Vietnam',
        'Russian Federation': 'Russia',
        'Republic of Korea': 'South Korea',
        'Republic of Moldova': 'Moldova',
        'China': 'Mainland China'
    }
    df.rename(country_rename, axis=0, inplace=True)

    # if 'Hong Kong SAR' in df.index:
    #     df.loc['Hong Kong'] = df.loc['Hong Kong'] + df.loc['Hong Kong SAR']
    # if 'Iran (Islamic Republic of)' in df.index:
    #    df.loc['Iran'] = df.loc['Iran'] + df.loc['Iran (Islamic Republic of)']
    # if 'Viet Nam' in df.index:
    #     df.loc['Vietnam'] = df.loc['Vietnam'] + df.loc['Viet Nam']
    # if 'Russian Federation' in df.index:
    #     df.loc['Russia'] = df.loc['Russia'] + df.loc['Russian Federation']
    # if 'Republic of Korea' in df.index:
    #     df.loc['South Korea'] = \
    #         df.loc['South Korea'] + df.loc['Republic of Korea']
    # if 'Republic of Moldova' in df.index:
    #     df.loc['Moldova'] = df.loc['Moldova'] + df.loc['Republic of Moldova']

    df.drop(constants.ignore_countries, axis=0, inplace=True, errors='ignore')


def _compute_days_since_nth_case(df_cases, n=1):
    ''' Compute the country-wise days since first confirmed case

    :param df_cases: country-wise time-series of confirmed case counts
    :return: Series of country-wise days since first case
    '''
    date_first_case = df_cases[df_cases >= n].idxmin(axis=1)
    days_since_first_case = date_first_case.apply(
        lambda x: 0 if pd.isnull(x) else (df_cases.columns.max() - x).days)
    # Add 1 month for China, since outbreak started late 2019:
    if 'Mainland China' in days_since_first_case.index:
        days_since_first_case.loc['Mainland China'] += 30
    # Fill in blanks (not yet reached n cases) with 0s:
    days_since_first_case.fillna(0, inplace=True)

    return days_since_first_case


def _add_cpi_data(df_input):
    '''
    Add the Government transparency (CPI - corruption perceptions index)
    data (by country) as a column in the COVID cases dataframe.

    :param df_input: COVID-19 data rolled up country-wise
    :return: None, add CPI data to df_input in place
    '''
    cpi_data = pd.read_excel(
        os.path.join(DATA_DIR, 'CPI2019.xlsx'), skiprows=2)
    cpi_data.set_index('Country', inplace=True, drop=True)
    cpi_data.rename(constants.cpi_country_mapping, axis=0, inplace=True)

    # Add CPI score to input df:
    df_input['cpi_score_2019'] = cpi_data['CPI score 2019']

    return df_input


def _add_wb_data(df_input):
    '''
    Add the World Bank data covariates as columns in the COVID cases dataframe.

    :param df_input: COVID-19 data rolled up country-wise
    :return: None, add World Bank data to df_input in place
    '''
    wb_data = pd.read_csv(
        os.path.join(DATA_DIR, 'world_bank_data.csv'),
        na_values='..')

    for (wb_name, var_name) in constants.wb_covariates:
        wb_series = wb_data.loc[wb_data['Series Code'] == wb_name]
        wb_series.set_index('Country Name', inplace=True, drop=True)
        wb_series.rename(constants.wb_country_mapping, axis=0, inplace=True)

        # Add WB data:
        df_input[var_name] = _get_most_recent_value(wb_series)


def _add_testing_data(df_input):

    df_tests = pd.read_csv(
        ('https://raw.githubusercontent.com/owid/owid-datasets/master/'
         'datasets/COVID-19%20Tests%20per%20million%20people/'
         'COVID-19%20Tests%20per%20million%20people.csv'),
        index_col='Entity')

    country_rename = {
        'United States - CDC samples tested': 'US',
        'China - Guangdong': 'Mainland China',
        'South Korea': 'Korea, South'
    }
    df_tests.rename(country_rename, axis=0, inplace=True)

    df_input['tests_per_million'] = \
        df_tests['Total COVID-19 tests performed per million people']


def _get_most_recent_value(wb_series):
    '''
    Get most recent non-null value for each country in the World Bank
    time-series data
    '''
    ts_data = wb_series[wb_series.columns[3::]]

    def _helper(row):
        row_nn = row[row.notnull()]
        if len(row_nn):
            return row_nn[-1]
        else:
            return np.nan

    return ts_data.apply(_helper, axis=1)


def _get_days_since_nth_state_case(state_abbrs, ts_cases, n,
                                   state_name_abbr_lookup):

    state_cases = [ts_cases.loc[ts_cases['Province/State'].apply(
        lambda x: (', %s' % sa in x) or
        (x == _state_name_lookup(state_name_abbr_lookup, sa)))].sum() for
        sa in state_abbrs]

    state_cases = pd.DataFrame(state_cases,
                               index=state_abbrs,
                               columns=state_cases[0].index)
    state_cases = state_cases[state_cases.columns[4::]]
    state_cases = _convert_cols_to_dt(state_cases)
    return _compute_days_since_nth_case(state_cases, n=n)


def _get_case_cts_n_days_ago(state_abbrs, ts_cases, n,
                             state_name_abbr_lookup):

    state_cases = [ts_cases.loc[ts_cases['Province/State'].apply(
        lambda x: (', %s' % sa in x) or
        (x == _state_name_lookup(state_name_abbr_lookup, sa)))].sum() for
        sa in state_abbrs]

    state_cases = pd.DataFrame(state_cases,
                               index=state_abbrs,
                               columns=state_cases[0].index)
    state_cases = state_cases[state_cases.columns[4::]]
    return state_cases[[state_cases.columns[-n]]]


def _state_name_lookup(state_name_abbr_lookup, state_abbr):
    return next(key for key, value in state_name_abbr_lookup.items()
                if value == state_abbr)
