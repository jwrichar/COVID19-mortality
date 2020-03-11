
ignore_countries = [
    'Others',
    'Andorra',
    'Saint Barthelemy',
    'Brunei',
    'Gibraltar',
    'Vatican City',
    'St. Martin',
    'Saint Martin',
    'Monaco',
    'Martinique',
    'Liechtenstein',
    'Faroe Islands',
    'Macau',
    'Palestine',
    'occupied Palestinian territory',
    'French Guiana',
    'Taiwan',
    'Taipei and environs',
    'San Marino',
    'Holy See',
    'Hong Kong SAR',
    'Iran (Islamic Republic of)',
    'Macao SAR',
    'Viet Nam',
    'Russian Federation',
    'Republic of Korea',
    'Republic of Moldova'
]

cpi_country_mapping = {
    'United States of America': 'US',
    'China': 'Mainland China',
    'United Kingdom': 'UK',
    'Korea, South': 'South Korea'
}


wb_covariates = [
    ('SH.XPD.OOPC.CH.ZS',
        'healthcare_oop_expenditure'),
    ('HD.HCI.OVRL',
        'hci'),
    # ('SH.MED.PHYS.ZS',
    #     'physicians_per_capita'),
    ('SP.POP.65UP.TO.ZS',
        'population_perc_over65'),
    ('SP.RUR.TOTL.ZS',
        'population_perc_rural')
]

wb_country_mapping = {
    'United States': 'US',
    'United Kingdom': 'UK',
    'Egypt, Arab Rep.': 'Egypt',
    'Hong Kong SAR, China': 'Hong Kong',
    'Iran, Islamic Rep.': 'Iran',
    'China': 'Mainland China',
    'Russian Federation': 'Russia',
    'Slovak Republic': 'Slovakia',
    'Korea, Rep.': 'South Korea'
}
