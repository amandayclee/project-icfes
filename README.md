# Columbia College Admissions Test Performance from 2014 to 2022 :microscope:

![version](https://img.shields.io/badge/version-1.0-blue) ![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg) ![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)

# Project Intro/Objective :mag:
The problem at hand revolves around the analysis of SAT scores in Colombia, particularly the Saber 11 exam administered by the Colombian Institute for the Evaluation of Education (ICFES). This standardized test plays a crucial role in the higher education admissions process in Colombia, with universities considering these scores along with other factors when admitting students.

The key objectives of this analysis can be outlined as follows:
1. Performance Analysis: Evaluate student performance trends over time and across different regions in Colombia. Identify strengths and weaknesses in the education system to highlight areas in need of improvement.
2. Socio Economic Impact: Explore the connections between socioeconomic factors in each department and SAT scores. Examine potential inequalities in education to inform initiatives for more equitable opportunities.
3. Predictive Modeling: Develop predictive models to forecast SAT scores based on socioeconomic features. Identify students who may require additional support, contributing to early interventions and improved academic outcomes.

# Remarks

This repo has the following basic structure.

```
├── environment.yml     <- Basic Python dependencies for Conda environment.
├── README.md           <- The top-level README for developers.
│
├── data                <- If data sets are too large for repo include in .gitignore and download/provide locally. If
│   │                      multiple data sets are used, create a seperate folder for each data set. Also data sets might
│   │                      be merged. Each subfolder (i.e. data stage) should contain a data description (e.g. URL etc.).
│   ├── 01_raw          <- Immutable input data
│   ├── 02_intermediate <- Cleaned version of raw (no missing values, outliers, unreadable data etc.)
│   ├── 03_processed    <- Train data used to develop models (including interactions, new features etc. with 2 columns
│   │                      (_a, _b) whenever there are transformations using a datapoint's label as part of feature),
│   │                      derived test data for prediction (using processed train data and _b columns when applicable)
│   ├── 04_models       <- Trained models (.pkl files using joblib). Naming convention is date YYYYMMDD (for ordering),
│   │                      '_', score, '_' and a short description of the used model
│   ├── 05_model_output <- Model output
│   └── 06_reporting    <- Reports and input to frontend
│
├── docs                <- Space for documentation. Can also included conceptualization and literature review.
│
├── references          <- Data dictionaries, manuals, reference manager (e.g. EndNote) etc.
│
├── results             <- Final analysis docs.
│   ├── figures         <- Generated graphics and figures to be used in reporting, presentations or papers
│   ├── presentations   <- Presentation slides (e.g. pptx) for conferences, seminars etc.
│   ├── submissions     <- Final submission files (e.g. csv, docx, pdf) including versioning (e.g. v1). Folder also
│   │                      contains revisions & resubmissions (create subfolders if applicable).
│   ├── tables          <- Generated tables to be used in reporting or papers
│
│
├── .gitignore          <- Avoids uploading data, credentials, outputs, system files etc.
│
├── src_py                    <- Python source code for use in this project.
│
├── src_r                     <- R source code for use in this project.
```


### Acknowledgment

Initial project structure was created following the structure in [this repo](https://github.com/malill/research-template).
