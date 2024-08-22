# %%
"""This notebook illustrates how to interactively create and stepwise test a pipeline that filters a given dataset based on the rule of five."""

#%% Imports
import pandas as pd
import os

from pdchemchain.links import PandasAddMoleculeColumn, RDKitDescriptors, Query, KeepColumns

#%% Getting a example dataset from an online source
csv_file = "SLC6A4_active_excape_export.csv"
if not os.path.exists(csv_file):
    import urllib.request
    url = "https://ndownloader.figshare.com/files/25747817"
    urllib.request.urlretrieve(url, csv_file)
data = pd.read_csv(csv_file)
df = data.sample(100)
df.head()

# %% First link is to add a molecular object column, and quickly test it. Alternatives to this link is MolFromSmiles link that work row by row, but gives better error messages.
makemols = PandasAddMoleculeColumn(smilesCol='SMILES')

makemols(df.sample(5)).head()
#The new column with RDKit molecular objects is seen most rightward

#%% Next is to calculate some needed descriptors. We begin to build our chain.
descriptors = RDKitDescriptors(descriptors=["MolWt", "MolLogP", "NumHAcceptors", "NumHDonors"])

chain = makemols + descriptors

chain(df.sample(5)).head()

# %% Then we filter with the Query Link and extend our chain
query = Query('MolWt < 500 and MolLogP < 5 and NumHAcceptors < 10  and NumHDonors < 5')

chain = chain + query

chain(df.sample(5)).head()

#%% Lastly we get rid of some of the columns we don't want to save
keep = KeepColumns(["Ambit_InchiKey", "SMILES","MolWt", "MolLogP", "NumHAcceptors", "NumHDonors"])
chain = chain + keep
chain(df.sample(5)).head()
# %% Now we can save the pipeline to a yaml file for reuse on the entire dataset
chain.to_config_file('Lipinski.yaml')

#%% A command line would then look like this
!pdchemchain run Lipinski.yaml --in_file SLC6A4_active_excape_export.csv --out_file Lipinski_filtered.csv # type: ignore 

