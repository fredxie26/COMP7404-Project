import pandas as pd

hospital_vent_icu_data = pd.read_csv("datasets/covid19-epiSummary-hospVentICU.csv")
public_infobase_data = pd.read_csv("datasets/covid19-public-infobase.csv")
vaccine_distribution_data = pd.read_csv("datasets/covid19-vaccination-distribution.csv")

# Only want to take total hospital occupancy as output label for rest of data from hospital_vent_icu_data
hospital_vent_icu_data.drop(['BEDCAP_ICU','VENTCAP_ICU','BEDCAP_OTHER','VENTCAP_OTHER',
'COVID_NEWICU','COVID_ICU','COVID_NEWOTHER','COVID_OTHER','COVID_VENT','VENTCAP_TOTAL','NONCOVID_ICU',
'NONCOVID_OTHER','NONCOVID_VENT'], axis=1, inplace=True)

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

# Remove any province specific data, we want the dataset to reflect canada-wide statistics only
merge_hospital_infobase_vaccine = merge_hospital_infobase_vaccine[merge_hospital_infobase_vaccine["prname"].str.contains("British Columbia") == False]
merge_hospital_infobase_vaccine = merge_hospital_infobase_vaccine[merge_hospital_infobase_vaccine["prname"].str.contains("Alberta") == False]
merge_hospital_infobase_vaccine = merge_hospital_infobase_vaccine[merge_hospital_infobase_vaccine["prname"].str.contains("Saskatchewan") == False]
merge_hospital_infobase_vaccine = merge_hospital_infobase_vaccine[merge_hospital_infobase_vaccine["prname"].str.contains("Manitoba") == False]
merge_hospital_infobase_vaccine = merge_hospital_infobase_vaccine[merge_hospital_infobase_vaccine["prname"].str.contains("Ontario") == False]
merge_hospital_infobase_vaccine = merge_hospital_infobase_vaccine[merge_hospital_infobase_vaccine["prname"].str.contains("Quebec") == False]
merge_hospital_infobase_vaccine = merge_hospital_infobase_vaccine[merge_hospital_infobase_vaccine["prname"].str.contains("Newfoundland and Labrador") == False]
merge_hospital_infobase_vaccine = merge_hospital_infobase_vaccine[merge_hospital_infobase_vaccine["prname"].str.contains("New Brunswick") == False]
merge_hospital_infobase_vaccine = merge_hospital_infobase_vaccine[merge_hospital_infobase_vaccine["prname"].str.contains("Nova Scotia") == False]
merge_hospital_infobase_vaccine = merge_hospital_infobase_vaccine[merge_hospital_infobase_vaccine["prname"].str.contains("Prince Edward Island") == False]
merge_hospital_infobase_vaccine = merge_hospital_infobase_vaccine[merge_hospital_infobase_vaccine["prname"].str.contains("Yukon") == False]
merge_hospital_infobase_vaccine = merge_hospital_infobase_vaccine[merge_hospital_infobase_vaccine["prname"].str.contains("Northwest Territories") == False]
merge_hospital_infobase_vaccine = merge_hospital_infobase_vaccine[merge_hospital_infobase_vaccine["prname"].str.contains("Nunavut") == False]

# now that all the entries are canada specific remove the prname column as it serves no purpose to distinguish rows by province.
merge_hospital_infobase_vaccine.drop(['prname'], axis=1, inplace=True)

# Removing any columns that contain all 1s or all 0s as they do not help the model to predict our goal
merge_hospital_infobase_vaccine.drop(['pruid_x'], axis=1, inplace=True)
merge_hospital_infobase_vaccine.drop(['update'], axis=1, inplace=True)
merge_hospital_infobase_vaccine.drop(['pruid_y'], axis=1, inplace=True)
merge_hospital_infobase_vaccine.drop(['updated'], axis=1, inplace=True)

# Print to csv
merge_hospital_infobase_vaccine.to_csv('combined-dataset.csv', index=False)



#df.to_csv('file_name.csv', index=False)
#df.to_csv('file_name.csv', encoding='utf-8')