# ML Data Preprocessing

This project shows a **simple way to prep data** before using any machine learning model.  
It’s meant to be easy to follow and learn from.

---

## What it does

Here’s what the code does:

- Handles missing values
- Encodes categorical (non-numeric) data
- Splits the dataset into training and testing sets
- Scales numerical columns so big numbers don’t mess up the model

---

## Dataset

The file `Data.csv` has:

- Country (France, Spain, Germany)
- Age
- Salary
- Purchased (Yes / No)

---

## How to use it

1. Put `Data.csv` in the same folder as `preprocessing.py`
2. Run the Python file
3. After running, you’ll get:
   - `x_train` and `x_test`: prepped features
   - `y_train` and `y_test`: encoded target labels

---

## Notes

- Some steps, like scaling, aren’t needed for certain models (like Decision Trees or Random Forest).  
- The code has comments explaining each step in a simple way.
