from mlsynth.mlsynth import FDID
import matplotlib.pyplot as plt
import pandas as pd
import os
from theme import jared_theme
import matplotlib

matplotlib.rcParams.update(jared_theme)

def get_edited_frames(stub_url, urls, base_dict):
    edited_frames = []

    for url, (key, params) in zip(urls, base_dict.items()):
        subdf = pd.read_csv(stub_url + url)

        # Keep only the specified columns
        subdf = subdf[params["Columns"]]

        # Ensure the time column is of integer type
        subdf[params["Time"]] = subdf[params["Time"]].astype(int)

        # Generate the treatment variable
        subdf[params["Treatment Name"]] = (
            subdf[params["Panel"]].str.contains(params["Treated Unit"])
        ) & (subdf[params["Time"]] >= params["Treatment Time"])

        # Handle specific case for Basque dataset
        if (
            key == "Basque"
            and "Spain (Espana)" in subdf[params["Panel"]].values
        ):
            subdf = subdf[
                ~subdf[params["Panel"]].str.contains("Spain \\(Espana\\)")
            ]
            subdf.loc[
                subdf["regionname"].str.contains("Vasco"), "regionname"
            ] = "Basque"

        # Append the edited DataFrame to the list
        edited_frames.append(subdf)

    return edited_frames


# Example usage
stub_url = "https://raw.githubusercontent.com/OscarEngelbrektson/SyntheticControlMethods/master/examples/datasets/"

base_dict = {
    "Basque": {
        "Columns": ["regionname", "year", "gdpcap"],
        "Treatment Time": 1975,
        "Treatment Name": "Terrorism",
        "Treated Unit": "Vasco",
        "Time": "year",
        "Panel": "regionname",
        "Outcome": "gdpcap",
    },
    "Germany": {
        "Columns": ["country", "year", "gdp"],
        "Treatment Time": 1978,
        "Treatment Name": "Reunification",
        "Treated Unit": "Germany",
        "Time": "year",
        "Panel": "country",
        "Outcome": "gdp",
    },
    "Smoking": {
        "Columns": ["state", "year", "cigsale"],
        "Treatment Time": 1989,
        "Treatment Name": "Proposition 99",
        "Treated Unit": "California",
        "Time": "year",
        "Panel": "state",
        "Outcome": "cigsale",
    },
}

edited_frames = get_edited_frames(
    stub_url,
    ["basque_data.csv", "german_reunification.csv", "smoking_data.csv"],
    base_dict,
)

number = 0
df = edited_frames[number]

# Get the keys as a list
keys_list = list(base_dict.keys())

# Match based on position
position = number  # For "Basque"
selected_key = keys_list[position]

# Access the corresponding dictionary
selected_dict = base_dict[selected_key]

# Example: Accessing specific values
columns = selected_dict["Columns"]
treatment_name = selected_dict["Treatment Name"]

# Example usage
unitid = df.columns[0]
time = df.columns[1]
outcome = df.columns[2]
treat = selected_dict["Treatment Name"]

new_directory = os.path.join(os.getcwd(), "examples")
os.chdir(new_directory)

# Define the 'FDID' directory
save_directory = os.path.join(os.getcwd(), "fdid")

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

save = {
    "filename": "FDID_Basque",
    "extension": "png",
    "directory": save_directory,
}

config = {
    "df": df,
    "treat": treat,
    "time": time,
    "outcome": outcome,
    "unitid": unitid,
    "counterfactual_color": "red",
    "treated_color": "black",
    "display_graphs": True,
    "save": save,
}

model = FDID(config)

# Run the FDID analysis
autores = model.fit()
