# kalbe-python-workshop-2025
## General Overview
Welcome to Kalbe Python Workshop 2025, this project is about (...), this project is aimed towards helping you solving (...) problems using Python, so make sure it is already installed


## Installation Guide
### 1  Choose your distribution
You can run the notebooks with **Anaconda**, **VS Code + Python extension**, or the minimalist **Miniforge** stack (Conda-compatible but smaller). Pick one:
#### 1.1  Anaconda:
1. Go to the [Anaconda Distribution download page](https://www.anaconda.com/products/distribution).
2. Select the installer matching your OS (Windows/macOS/Linux) and download it.
3. Install it and Launch the App

#### 1.2  VS Code:
1. Install [Visual Studio Code](https://code.visualstudio.com/Download) for your OS.  
2. Open VS Code → **Extensions** sidebar → install the **Python** extension (by Microsoft).  
3. *(Windows only)* install the [Windows Terminal](https://aka.ms/terminal) for a nicer shell.  
4. Start VS Code, press <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd> → **Python: Select Interpreter** → choose your conda environment or base Python.  


#### 1.3  Miniforge:
Miniforge gives you a **conda‑compatible** environment without the 3 GB Anaconda payload.
```bash
# macOS / Linux
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-x86_64.sh -o miniforge.sh
bash miniforge.sh  # follow the prompts
source ~/miniforge3/bin/activate

# Windows (PowerShell)
Invoke-WebRequest -Uri https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe -OutFile miniforge.exe
./miniforge.exe  # run the installer
```

### 2  Install JupyterLab (all methods):
```bash
pip install --upgrade pip
pip install jupyterlab
jupyter lab  # launches in your default browser
```

### 3 Environment Setup
1. We recommend to create environment for each of your projects if they will likely mess up library version and dependencies (you could have an environment for general EDA, one for deep learning, one for optimization, and so on)
2. To create an environment, you can open your chosen terminal (or anaconda prompt), and do: ```conda create --name my_env_name python=3.11```
3. You then can activate that using ```conda activate myenv```

### 4 Cloning the Repo
1. You then can navigate to your prepared folder ```cd /path/to/your/project```
2. clone this repo ```git clone https://github.com/faathirchikal/kalbe-python-workshop-2025.git```
3. Navigate to the project folder
4. Install the libraries ```pip install -r requirements.txt -U```
5. Done, you can navigate the notebook as you wish

## Data Overview
This project data is about retail sales for each product category with additional information like discount, promotion, competitor pricing etc. you can download it from here: [Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset)
Data definition:
| Column               | Type      | Description |
|----------------------|-----------|-------------|
| `Date`               | object      | Daily records from start_date to end_date |
| `Store ID`           | object  | Outlet identifier |
| `Product ID`         | object  | SKU identifier |
| `Category`           | object  | Product categories like Electronics, Clothing, Groceries, etc. |
| `Region`             | object  | Geographic region of the store |
| `Inventory Level`             | object  | Stock available at the beginning of the day |
| `Units Sold`         | int64       | Units sold during the day |
| `Units Ordered`      | int64       |  |
| `Demand Forecast`      | int64       | Predicted demand based on past trends |
| `Price`              | float     |  |
| `Discount`           | float     | % discount applied (0‑100) |
| `Weather Condition`  | category  | Daily weather impacting sales |
| `Holiday/Promotion`  | bool      | `True` if holiday/promo active |
| `Competitor Pricing` | float     |  |
| `Seasonality`        | category  | Season label (e.g. Autumn/Winter) |

## Project Structure

- Data:
- 01_data_preprocessing.ipynb
- 02_eda.ipbyb
- 03_modelling_or_optimization.ipynb
- 04_prescriptive_and_streamlit.ipynb