import pandas as pd
import datetime

# Function to insert row in a dataframe at desired location based from --> https://www.geeksforgeeks.org/insert-row-at-given-position-in-pandas-dataframe/
def Insert_row(row_number, df, new_entries_to_insert):
    # Slice the upper half of the dataframe
    upper = df[0:row_number]
    upper.to_csv('test-upper.csv', index=False)
    # Store the result of lower half of the dataframe
    lower = df[row_number:]
    lower.to_csv('test-lower.csv', index=False)
    exit()
  
    # Insert the row in the upper half dataframe
    #pd.concat([upper, new_entries_to_insert])
  
    # Concat the two dataframes
    #df_result = pd.concat([upper, lower])
  
    # Reassign the index labels
    #df_result.index = [*range(df_result.shape[0])]
  
    # Return the updated dataframe
    #return df_result

hospital_vent_icu_data = pd.read_csv("datasets/covid19-epiSummary-hospVentICU.csv")
public_infobase_data = pd.read_csv("datasets/covid19-public-infobase.csv")
vaccine_distribution_data = pd.read_csv("datasets/covid19-vaccination-distribution.csv")

# Only want to take total hospital occupancy as output label for rest of data from hospital_vent_icu_data
hospital_vent_icu_data.drop(['BEDCAP_ICU','VENTCAP_ICU','BEDCAP_OTHER','VENTCAP_OTHER',
'COVID_NEWICU','COVID_ICU','COVID_NEWOTHER','COVID_OTHER','COVID_VENT','VENTCAP_TOTAL','NONCOVID_ICU',
'NONCOVID_OTHER','NONCOVID_VENT'], axis=1, inplace=True)

# Remove any province specific data, we want the final dataset to reflect canada-wide statistics only
public_infobase_data = public_infobase_data[public_infobase_data["prname"].str.contains("British Columbia") == False]
public_infobase_data = public_infobase_data[public_infobase_data["prname"].str.contains("Alberta") == False]
public_infobase_data = public_infobase_data[public_infobase_data["prname"].str.contains("Saskatchewan") == False]
public_infobase_data = public_infobase_data[public_infobase_data["prname"].str.contains("Manitoba") == False]
public_infobase_data = public_infobase_data[public_infobase_data["prname"].str.contains("Ontario") == False]
public_infobase_data = public_infobase_data[public_infobase_data["prname"].str.contains("Quebec") == False]
public_infobase_data = public_infobase_data[public_infobase_data["prname"].str.contains("Newfoundland and Labrador") == False]
public_infobase_data = public_infobase_data[public_infobase_data["prname"].str.contains("New Brunswick") == False]
public_infobase_data = public_infobase_data[public_infobase_data["prname"].str.contains("Nova Scotia") == False]
public_infobase_data = public_infobase_data[public_infobase_data["prname"].str.contains("Prince Edward Island") == False]
public_infobase_data = public_infobase_data[public_infobase_data["prname"].str.contains("Yukon") == False]
public_infobase_data = public_infobase_data[public_infobase_data["prname"].str.contains("Northwest Territories") == False]
public_infobase_data = public_infobase_data[public_infobase_data["prname"].str.contains("Nunavut") == False]
public_infobase_data = public_infobase_data[public_infobase_data["prname"].str.contains("Repatriated travellers") == False]
# do the same for the vaccine distribution set as well
vaccine_distribution_data = vaccine_distribution_data[vaccine_distribution_data["prname"].str.contains("British Columbia") == False]
vaccine_distribution_data = vaccine_distribution_data[vaccine_distribution_data["prname"].str.contains("Alberta") == False]
vaccine_distribution_data = vaccine_distribution_data[vaccine_distribution_data["prname"].str.contains("Saskatchewan") == False]
vaccine_distribution_data = vaccine_distribution_data[vaccine_distribution_data["prname"].str.contains("Manitoba") == False]
vaccine_distribution_data = vaccine_distribution_data[vaccine_distribution_data["prname"].str.contains("Ontario") == False]
vaccine_distribution_data = vaccine_distribution_data[vaccine_distribution_data["prname"].str.contains("Quebec") == False]
vaccine_distribution_data = vaccine_distribution_data[vaccine_distribution_data["prname"].str.contains("Newfoundland and Labrador") == False]
vaccine_distribution_data = vaccine_distribution_data[vaccine_distribution_data["prname"].str.contains("New Brunswick") == False]
vaccine_distribution_data = vaccine_distribution_data[vaccine_distribution_data["prname"].str.contains("Nova Scotia") == False]
vaccine_distribution_data = vaccine_distribution_data[vaccine_distribution_data["prname"].str.contains("Prince Edward Island") == False]
vaccine_distribution_data = vaccine_distribution_data[vaccine_distribution_data["prname"].str.contains("Yukon") == False]
vaccine_distribution_data = vaccine_distribution_data[vaccine_distribution_data["prname"].str.contains("Northwest Territories") == False]
vaccine_distribution_data = vaccine_distribution_data[vaccine_distribution_data["prname"].str.contains("Nunavut") == False]
vaccine_distribution_data = vaccine_distribution_data[vaccine_distribution_data["prname"].str.contains("Federal allocation") == False]

# Before we merge the datasets based on date we have to adjust the weekly averages in the public infobase to be daily instead
# so we have more data to work with from the other datasets after merging instead of being restricted by weekly dates from the public infobase.
# Attempting to do so by slicing when a new weekly entry date is encountered, generating 6 more entries with the same row information 
# but with adjusted date according to the 6 days preceding the original average entry in order to mimick daily entries of the averaged data.

# Set a minimum date to begin comparison against as iteration continues
lastdate = datetime.datetime.min
#public_infobase_data.to_csv('test-test.csv', index=False)

# We can use iterrows to iterate over a dataframe by row
for num, row in public_infobase_data.iterrows():
    # Extract date from current row into a datetime object, date is the 4th item in a row
    currdate = datetime.datetime.strptime(row[3], '%Y-%m-%d')
    #print("CURR DATE: ", currdate, " ----------------------------------------------------------------------------------------------------")
    # Check if we found a new date while iterating, this is the starting point to make 6 preceding days to represent the full week of daily entries
    if currdate >= lastdate:
        # update the lastdate value for next loop
        lastdate = currdate
        # Generate 6 entries based on currdate and store them in a temporary list to be converted to a dataframe and added to the original dataset
        templist = []
        N = 6
        for n in reversed(range(N)):
            n += 1
            temprow = row.copy(deep=True)
            # temprow[3] is the entry date of the row in string format, need to change to datetime to apply day adjustment then back to string
            tempdate = datetime.datetime.strptime(temprow[3], '%Y-%m-%d')
            tempdate = tempdate - datetime.timedelta(days=n) # adjust the date relative to the original to represent preceding dates
            temprow[3] = tempdate.strftime('%Y-%m-%d') # back to string
            templist.append(temprow) # save to list to be injected into dataframe later
        
        # We now have a list of pandas series objects, with each object representing a row for a daily entry preceding the original entry
        new_daily_entries = pd.DataFrame(templist)
        # add the new entries into the original dataset.

public_infobase_data.to_csv('test-post-daily-adjust.csv', index=False)

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