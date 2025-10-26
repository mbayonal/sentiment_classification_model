# IMDB Dataset DVC Repository

This repository contains the IMDB dataset managed with Data Version Control (DVC). The dataset includes information about movies, TV shows, and other media from the Internet Movie Database.

## Dataset Description

The dataset consists of several TSV files containing different aspects of the IMDB database:

- **title.akas.tsv.gz**: Alternative titles for media
  - titleId (string) - a tconst, an alphanumeric unique identifier of the title
  - ordering (integer) – a number to uniquely identify rows for a given titleId
  - title (string) – the localized title
  - region (string) - the region for this version of the title
  - language (string) - the language of the title
  - types (array) - Enumerated set of attributes for this alternative title
  - attributes (array) - Additional terms to describe this alternative title
  - isOriginalTitle (boolean) – 0: not original title; 1: original title

- **title.basics.tsv.gz**: Basic information about titles
  - tconst (string) - alphanumeric unique identifier of the title
  - titleType (string) – the type/format of the title
  - primaryTitle (string) – the more popular title
  - originalTitle (string) - original title, in the original language
  - isAdult (boolean) - 0: non-adult title; 1: adult title
  - startYear (YYYY) – represents the release year of a title
  - endYear (YYYY) – TV Series end year
  - runtimeMinutes – primary runtime of the title, in minutes
  - genres (string array) – includes up to three genres associated with the title

- **title.crew.tsv.gz**: Directors and writers for titles
  - tconst (string) - alphanumeric unique identifier of the title
  - directors (array of nconsts) - director(s) of the given title
  - writers (array of nconsts) – writer(s) of the given title

- **title.episode.tsv.gz**: TV episode information
  - tconst (string) - alphanumeric identifier of episode
  - parentTconst (string) - alphanumeric identifier of the parent TV Series
  - seasonNumber (integer) – season number the episode belongs to
  - episodeNumber (integer) – episode number of the tconst in the TV series

- **title.principals.tsv.gz**: Principal cast/crew for titles
  - tconst (string) - alphanumeric unique identifier of the title
  - ordering (integer) – a number to uniquely identify rows for a given titleId
  - nconst (string) - alphanumeric unique identifier of the name/person
  - category (string) - the category of job that person was in
  - job (string) - the specific job title if applicable, else '\N'
  - characters (string) - the name of the character played if applicable, else '\N'

- **title.ratings.tsv.gz**: User ratings for titles
  - tconst (string) - alphanumeric unique identifier of the title
  - averageRating – weighted average of all the individual user ratings
  - numVotes - number of votes the title has received

- **name.basics.tsv.gz**: Information about individuals
  - nconst (string) - alphanumeric unique identifier of the name/person
  - primaryName (string)– name by which the person is most often credited
  - birthYear – in YYYY format
  - deathYear – in YYYY format if applicable, else '\N'
  - primaryProfession (array of strings)– the top-3 professions of the person
  - knownForTitles (array of tconsts) – titles the person is known for

## Repository Structure

```
.
├── data/
│   ├── raw/       # Raw data files
│   └── processed/  # Processed data files
├── models/         # Trained models
├── notebooks/      # Jupyter notebooks
├── src/            # Source code
│   ├── data/       # Scripts for data processing
│   ├── features/   # Scripts for feature engineering
│   └── models/     # Scripts for model training
├── .dvc/           # DVC configuration
├── .gitignore      # Git ignore file
├── dvc.yaml        # DVC pipeline definition
├── requirements.txt # Python dependencies
└── README.md       # This file
```

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Initialize DVC: `dvc init`
4. Add data: `dvc add data/raw/<file>`
5. Pull the data: `dvc pull`

## DVC Pipeline

The DVC pipeline consists of the following stages:

1. Data download
2. Data preprocessing
3. Feature extraction
4. Model training
5. Model evaluation

To run the entire pipeline:

```
dvc repro
```

To run a specific stage:

```
dvc repro <stage_name>
```
