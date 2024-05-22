# mlsynth
This is the repository containing the code for the Python library **mlsynth**. I provide detailed tutorials for each estimator in the package. As of May 5, 2024, we have the two-step SCM, linking to my Forward DID package (which will someday be moved here and integrated), as well as the Principal Component Regression estimator for synthetic control methods.

To use **mlsynth**, *at minimum* you need a Python dataframe with

- One column for the Unit ID (these must be unique, but they may be string values or numerical values).
- One column for time (indexed to the unit variable, these must form a balanced panel).
- A numerical outcome column of interest, and
- One column that is an indicator denoting treatment status. At this time, **mlsynth** only accepts binary interventions and one treated unit per df (this does NOT MEAN that you can't use the current functionality in settings of staggered adoption, but it *does mean* that you, the user, must be mindful about things like event-time and averaging your ATTs accordingly). In time, staggered adoption settings will come.
