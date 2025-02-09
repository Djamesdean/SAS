import json
import pandas as pd
import great_expectations as gx
import mlflow

df = pd.read_csv("/Users/merkava/Documents/School/Software engineering/SAS/data/raw/Steam.csv")
context = gx.get_context()
data_source = context.get_datasource("pandas")
data_asset = data_source.add_dataframe_asset(name="steam_dataframe1")

batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")
batch = batch_definition.get_batch(batch_parameters={"dataframe": df})




# Add expectations to the suite

expectation1 = gx.expectations.ExpectColumnDistinctValuesToBeInSet(column="user_suggestion",value_set=[1, 0])
expectation2 = gx.expectations.ExpectColumnValuesToNotBeNull(column="user_review")



validation_result1 = batch.validate(expectation1)
validation_result2 = batch.validate(expectation2)

print(validation_result1)
print(validation_result2)


with open("validation_results.json", "w") as f:
    json.dump(validation_result1.to_json_dict(), f, indent=4)
with open("validation_results.json", "w") as f:
    json.dump(validation_result2.to_json_dict(), f, indent=4)