import pandas as pd

hospital_vent_icu_data = pd.read_csv("datasets/covid19-epiSummary-hospVentICU.csv")
public_infobase_data = pd.read_csv("datasets/covid19-public-infobase.csv")
vaccine_distribution_data = pd.read_csv("datasets/covid19-vaccination-distribution.csv")

# Merging the data sets based on date, only entires that share a date will exist in the final merged set, those that do not will be dropped
merge_hospital_infobase = pd.merge(hospital_vent_icu_data, public_infobase_data, on =['date'])
merge_hospital_infobase_vaccine = pd.merge(merge_hospital_infobase, vaccine_distribution_data, on =['date', 'prname'])

# Remove the vaccination variant count columns as they are mostly missing values from the original dataset
# We will be using the total vaccination rate instead.
merge_hospital_infobase_vaccine.drop(['numdelta_all_distributed','numdelta_pfizerbiontech_distributed',
'numdelta_pfizerbiontech_5_11_distributed','numdelta_moderna_distributed','numdelta_astrazeneca_distributed',
'numdelta_janssen_distributed','numdelta_novavax_distributed'], axis=1, inplace=True)

# Also dropping french province names, we don't need em
merge_hospital_infobase_vaccine.drop(['prnameFR'], axis=1, inplace=True)

# Esnure to remove any rows that contain nan values so we have a populated dataset to work with remaining after.
merge_hospital_infobase_vaccine.dropna(axis=0, how='any', inplace=True)
merge_hospital_infobase_vaccine.to_csv('combined-dataset.csv')



#df.to_csv('file_name.csv', index=False)
#df.to_csv('file_name.csv', encoding='utf-8')