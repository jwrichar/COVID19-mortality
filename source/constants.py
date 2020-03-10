
ignore_countries = [
    'Others',
    'Andorra',
    'Saint Barthelemy',
    'Brunei',
    'Gibraltar',
    'Vatican City',
    'St. Martin',
    'Monaco',
    'Martinique',
    'Liechtenstein',
    'Faroe Islands',
    'Macau',
    'Palestine',
    'French Guiana',
    'Taiwan',
    'San Marino'
]

cpi_country_mapping = {
    'United States of America': 'US',
    'China': 'Mainland China',
    'United Kingdom': 'UK',
    'Korea, South': 'South Korea'
}


wb_covariates = [
    ('SH.XPD.CHEX.PC.CD',
        'healthcare_spending_per_capita'),
    ('HD.HCI.OVRL',
        'hci'),
    ('SH.MED.PHYS.ZS',
        'physicians_per_capita'),
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
