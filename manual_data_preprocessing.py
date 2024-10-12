import pandas as pd

# Use the specified path for the CSV file
studydata = pd.read_csv(r'C:\Users\Laurenz\Documents\00AA UNI\3. Semester\Machine Learning - Unsupervised Learning and Feature Engineering\Project\Data\mental-heath-in-tech-2016_20161114.csv')

# Display the first few rows of the DataFrame
print(studydata.head())

# Display all column labels of the DataFrame
print(studydata.columns)

# Create a dictionary with column indices as keys and column names as values
column_index_mapping = {column: index for index, column in enumerate(studydata.columns)}

# Display the dictionary
print(column_index_mapping)


column_name_abbreviations = ['self_emp',
                 'company_size',
                 'tech_employer',
                 'tech_role',
                 'mental_health_benefits',
                 'aware_mental_health_options',
                 'discussed_mental_health',
                 'mental_health_resources',
                 'anonymity_mental_health',
                 'mental_health_leave_comfort',
                 'neg_mental_health_conseq_employer',
                 'neg_physical_health_conseq_employer',
                 'comfortable_with_coworkers',
                 'comfortable_with_supervisor',
                 'employer_takes_mental_health_seriously',
                 'neg_consequences_mental_health',
                 'medical_coverage_mental_health',
                 'aware_local_resources',
                 'reveal_to_clients',
                 'neg_conseq_revealed_client',
                 'reveal_to_coworkers',
                 'neg_conseq_revealed_coworker',
                 'mental_health_affects_productivity',
                 'work_time_affected_by_mental_health',
                 'previous_employers',
                 'previous_employers_mental_health_benefits',
                 'aware_prev_mental_health_options',
                 'prev_discussed_mental_health',
                 'prev_provided_resources',
                 'prev_anonymity_protected',
                 'prev_neg_mental_health_conseq',
                 'prev_neg_physical_health_conseq',
                 'discuss_with_prev_coworkers',
                 'discuss_with_prev_supervisor',
                 'prev_employers_takes_mental_health_seriously',
                 'prev_negative_consequences',
                 'physical_health_with_potential_employer',
                 'why_not_physical_health',
                 'mental_health_with_potential_employer',
                 'why_not_mental_health',
                 'mental_health_hurt_career',
                 'team_view_negatively',
                 'share_with_family',
                 'observed_bad_handling',
                 'observed_bad_handling_effect',
                 'family_history_mental_health',
                 'past_mental_health_disorder',
                 'current_mental_health_disorder',
                 'diagnosed_conditions',
                 'suspected_conditions',
                 'diagnosed_by_professional',
                 'conditions_diagnosed_by_professional',
                 'sought_mental_health_treatment',
                 'mental_health_treated_interferes',
                 'mental_health_not_treated_interferes',
                 'age', 'gender',
                 'country_live',
                 'state_live',
                 'country_work',
                 'state_work',
                 'work_position',
                 'remote_work']

print("column_index_mapping dtype is: " ,type(column_index_mapping))
print("column_name_abbreviations type is: ", type(column_name_abbreviations))

print("column_index_mapping size is: " ,len(column_index_mapping))
print("column_name_abbreviations size is: " ,len(column_name_abbreviations))

if len(column_index_mapping) == len(column_name_abbreviations):
    print("The two lists have the same length.")
    
    # Merge abbreviations with column names dictionary
    
    print("/n/nOriginal column_index_mapping:")
    for key in column_index_mapping:
            print("column_index_mapping[key]:", column_index_mapping[key])
            print("column_name_abbreviations[column_index_mapping[key]]:", column_name_abbreviations[column_index_mapping[key]])
            column_index_mapping[key] = column_name_abbreviations[column_index_mapping[key]]

            

    print("The dictionary has been successfully merged.")
    print(" \n\n Updated column_index_mapping:", column_index_mapping, "\n\n ")

    
else:
    print("The two lists have different lengths.")


print("all done5")