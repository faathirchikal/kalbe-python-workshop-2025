# kalbe-python-workshop-2025
## General Overview
Welcome to Kalbe Python Workshop 2025, this project is about (...), this project is aimed towards helping you solving (...) problems using Python, so make sure it is already installed


## Installation Guide
### Choosing your App
Python can be installed through many ways, you could do it through Anaconda / VS Code / Miniforge
For Anaconda:
1. Go to the [Anaconda Distribution download page](https://www.anaconda.com/products/distribution).
2. Select the installer matching your OS (Windows/macOS/Linux) and download it.
3. Install it and Launch the App

For VS Code:

For Miniforge:

Installing jupyterlab:
```pip install --upgrade pip```
```pip install jupyterlab```
```jupyter lab```

### Environment Setup
1. We recommend to create environment for each of your projects if they will likely mess up library version and dependencies (you could have an environment for general EDA, one for deep learning, one for optimization, and so on)
2. To create an environment, you can open your chosen terminal (or anaconda prompt), and do: ```conda create --name my_env_name python=3.11```
3. You then can activate that using ```conda activate myenv```

### Cloning the Repo
1. You then can navigate to your prepared folder ```cd /path/to/your/project```
2. clone this repo ```git clone https://github.com/faathirchikal/kalbe-python-workshop-2025.git`
3. Navigate to the project folder
4. Install the libraries ```pip install -r requirements.txt -U```
5. Done, you can navigate the notebook as you wish

## Data Overview
This project data is about (...), you can download it from here: (...)
Data definition:
- column_a: ...

## Project Structure

- Data:
- 01_data_preprocessing.ipynb
- 02_eda.ipbyb
- 03_modelling_or_optimization.ipynb
- 04_prescriptive_and_streamlit.ipynb