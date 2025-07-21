# kalbe-python-workshop-2025
## General Overview
Welcome to the Kalbe Python Workshop 2025! This project focuses on Exploratory Data Analysis (EDA) using the [Online Retail II dataset](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci). The workshop is designed to help participants solve real-world data analysis problems using Python, so make sure it is already installed


## Installation Guide
### 1  Choose your distribution
You can run the notebooks using one of the following environments:
- **Anaconda**
- **VS Code + Python extension**
- **Miniforge** (lightweight Conda alternative)
  
#### 1.1  Anaconda:
1. Go to the [Anaconda Distribution download page](https://www.anaconda.com/products/distribution).
2. Select the installer matching your OS (Windows/macOS/Linux) and download it.
3. Install it and Launch the App
   

#### 1.2  VS Code:
1. Download and Install [Visual Studio Code](https://code.visualstudio.com/Download) for your OS.  
2. Open VS Code → **Extensions** sidebar → install the **Python** extension (by Microsoft).  
3. *(Windows only)* install the [Windows Terminal](https://aka.ms/terminal) for a nicer shell.  
4. Start VS Code, press <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd> → **Python: Select Interpreter** → choose your conda environment or base Python.  


#### 1.3  Miniforge:
Miniforge provides a lightweight Conda-compatible environment.
1. Go to the [Miniforge](https://github.com/conda-forge/miniforge/releases/tag/25.3.0-3)
2. Select the installer matching your OS (Windows/macOS/Linux) and download it.
3. Install it and Launch Miniforge Prompt

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
This project data is about retail sales for each product category with additional information like discount, promotion, competitor pricing etc. you can download it from here: [Online Retail II UC](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) and put them in data/raw/ folder
Data definition:
| Column               | Type      | Description |
|----------------------|-----------|-------------|
| `InvoiceNo`               | object      | Invoice number. Nominal. A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'c', it indicates a cancellation |
| `StockCode`           | object  | Product (item) code. Nominal. A 5-digit integral number uniquely assigned to each distinct product |
| `Description`         | object  | Product (item) name. Nominal |
| `Quantity`           | int64  | The quantities of each product (item) per transaction. Numeric |
| `InvoiceDate`             | datetime  | Invoice date and time. Numeric. The day and time when a transaction was generated |
| `UnitPrice`             | float64  | Unit price. Numeric. Product price per unit in sterling (£) |
| `CustomerID`         | object       | Customer number. Nominal. A 5-digit integral number uniquely assigned to each customer |
| `Country`      | object       | Country name. Nominal. The name of the country where a customer resides |


## Project Structure

```
├── data/
│   └── preprocessed/            # preprocessed data folder
│   └── raw/                     # raw data folder
├── 01_data_preprocessing.py     # Data Cleaning            
├── 02_eda.py                    # General EDA
├── 03A_price_elasticity.py      # Product Price Elasticity
├── 03B_other_analysis.py        # Other Analysis
├── 04_forecast_preprocessing.py # Preprocess for forecast
├── 05_forecast.py               # Forecast code
├── requirements.txt             # Libraries needed
├── streamlit_app.py             # Streamlit app
```
- Data:
- 01_data_preprocessing.ipynb
- 02_eda.ipbyb
- 03_modelling_or_optimization.ipynb
- 04_prescriptive_and_streamlit.ipynb
