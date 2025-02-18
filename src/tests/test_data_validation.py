import json
import pandas as pd
import great_expectations as gx

# Load your data
df = pd.read_csv("/Users/merkava/Documents/School/Software engineering/SAS/data/raw/Steam.csv")

# Set up Great Expectations context
context = gx.get_context()
data_source = context.get_datasource("pandas")
data_asset = data_source.add_dataframe_asset(name="steam_dataframe101")

# Create a batch for the dataframe
batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")
batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

# Add expectations
expectation1 = gx.expectations.ExpectColumnDistinctValuesToBeInSet(column="user_suggestion", value_set=[1, 0])
expectation2 = gx.expectations.ExpectColumnValuesToNotBeNull(column="user_review")

# Validate both expectations
validation_result1 = batch.validate(expectation1)
validation_result2 = batch.validate(expectation2)

# Print results
print("Validation result for 'user_suggestion' column:")
print(validation_result1)
print("\nValidation result for 'user_review' column:")
print(validation_result2)

# Save both results into different files
with open("validation_result_user_suggestion.json", "w") as f1:
    json.dump(validation_result1.to_json_dict(), f1, indent=4)

with open("validation_result_user_review.json", "w") as f2:
    json.dump(validation_result2.to_json_dict(), f2, indent=4)