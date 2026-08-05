"""User-editable settings for DigiAra inference.

Edit only the values in this file. The other scripts locate all model and data
files from the DigiAra project folder automatically.
"""

# Choose "KO" for knockout or "OE" for overexpression.
PERTURBATION_TYPE = "KO"

# Enter one or more Arabidopsis gene locus IDs. Multiple genes are supported.
MUTANT_GENES = ["AT4G33430", "AT2G13790"]

# This label is used in output filenames. Leave it empty to use the gene IDs.
MUTANT_NAME = "bak1_bkk1"

# Supported values: "PstDC3000" and "mock".
MICROBE = "PstDC3000"
