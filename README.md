# Week 1 - Machine Learning & AI Internship

## Mini Project
**Titanic Survival Prediction - Data Cleaning Project**

This project follows the Week 1 assignment requirements:
- Load and explore a dataset using Pandas
- Use `.info()` and `.describe()`
- Handle missing data
- Encode `Sex` with `LabelEncoder`
- Encode `Embarked` with `OneHotEncoder`
- Visualize age distribution with Matplotlib
- Export the cleaned dataset as a new CSV

## Project structure

```text
week1_ml_ai_assignment/
├── data/
│   └── Titanic-Dataset.csv        # add the Kaggle Titanic CSV here
├── outputs/
│   ├── titanic_cleaned.csv        # generated after running
│   └── age_distribution.png       # generated after running
├── week1_assignment.py
├── week1_practice.py
├── requirements.txt
└── README.md
```

## Setup

Open the project folder in VS Code.

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\Activate.ps1
```

Install libraries:

```bash
pip install -r requirements.txt
```

## Dataset

Download the **Titanic Dataset (Kaggle)** and place the CSV in the `data` folder with this exact name:

```text
Titanic-Dataset.csv
```

The Week 1 course PDF specifies the Titanic Dataset (Kaggle) and asks to clean missing data, encode `Sex` and `Embarked`, visualize age distribution, and output a cleaned CSV.

## Run the mini project

```bash
python week1_assignment.py
```

The generated files will appear inside `outputs/`.

## Run the practice set

```bash
python week1_practice.py
```

The practice PDF asks students to load a CSV, split data into train/test, train Linear Regression, and predict house price from area.

## GitHub submission

After testing the project:

```bash
git init
git add .
git commit -m "Complete Week 1 ML AI internship assignment"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Then submit the GitHub repository URL in the Skill Nexis Week 1 submission box.
