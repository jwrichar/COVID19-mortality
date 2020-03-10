
import pymc3 as pm


def initialize_model(df):

    # Normalize input covariates in a way that is sensible:

    # (1) days since first case: upper
    # mu_0 to reflect asymptotic mortality rate months after outbreak
    _normalize_col(df, 'days_since_first_case', how='upper')
    # (2) healthcare spending: mean
    # could also be upper, but for now take agnostic approach
    _normalize_col(df, 'healthcare_spending_per_capita', how='mean')
    # (3) hci = human capital index: upper
    # HCI measures education/health; mu_0 should reflect best scenario
    _normalize_col(df, 'hci', how='upper')
    # (4) % over 65: mean
    # mu_0 to reflect average world demographic
    _normalize_col(df, 'population_perc_over65', how='mean')
    # (5) CPI score: upper
    # mu_0 to reflect scenario in absence of corrupt govts
    _normalize_col(df, 'cpi_score_2019', how='upper')

    n = len(df)

    covid_mortality_model = pm.Model()

    with covid_mortality_model:

        # Priors:
        mu_0 = pm.Beta('mu_0', alpha=0.1, beta=10)
        sig_0 = pm.Uniform('sig_0', lower=0.0, upper=mu_0 * (1 - mu_0))
        # beta = pm.Normal('beta', mu=0, sigma=10, shape=5)
        # sigma = pm.HalfNormal('sigma', sigma=1)

        # Model mu from country-wise covariates:
        # mu_est = mu_0 + \
        #     beta[0] * df['days_since_first_case_normalized'].values + \
        #     beta[1] * df['healthcare_spending_per_capita_normalized'].values + \
        #     beta[2] * df['hci_normalized'].values + \
        #     beta[3] * df['population_perc_over65_normalized'].values + \
        #     beta[4] * df['cpi_score_2019_normalized'].values
        # mu_model = pm.Normal('mu_model', mu=mu_est, sigma=sigma, shape=len(df))

        # tau_i, mortality rate for each country
        # Parametrize with (mu, kappa) (e.g. mean, concentration)
        # instead of (alpha, beta) to ease interpretability.
        # alpha = mu*kappa
        # beta = (1 - mu)*kappa
        # tau = pm.Beta('tau', mu=mu_model, sigma=kappa, shape=len(df))
        tau = pm.Beta('tau', mu=mu_0, sigma=sig_0, shape=n)

        # Binomial likelihood:
        d_obs = pm.Binomial('d_obs',
                            n=df['cases'].values,
                            p=tau,
                            observed=df['deaths'].values)


    return covid_mortality_model


def _normalize_col(df, colname, how='mean'):
    '''
    Normalize an input column in one of 3 ways:

    * how=mean: unit normal N(0,1)
    * how=upper: normalize to [-1, 0] with highest value set to 0
    * how=lower: normalize to [0, 1] with lowest value set to 0

    Returns df modified in place with extra column added.
    '''
    colname_new = '%s_normalized' % colname
    if how == 'mean':
        mu = df[colname].mean()
        sig = df[colname].std()
        df[colname_new] = (df[colname] - mu) / sig
    elif how == 'upper':
        maxval = df[colname].max()
        minval = df[colname].min()
        df[colname_new] = (df[colname] - maxval) / (maxval - minval)
    elif how == 'lower':
        maxval = df[colname].max()
        minval = df[colname].min()
        df[colname_new] = (df[colname] - minval) / (maxval - minval)
