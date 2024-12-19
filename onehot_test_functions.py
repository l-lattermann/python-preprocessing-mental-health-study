import pandas as pd 

with open("data/mh_professional_diagnosis.csv") as file:
    df = pd.read_csv(file)


print(type(df['mh_professional_diagnosis'].iloc[0]))

# Explode the list column into individual rows
df_exploded = df.explode("mh_professional_diagnosis")





