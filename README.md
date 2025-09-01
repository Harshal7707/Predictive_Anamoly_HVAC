

# Predictive Anomaly HVAC

A Python project to detect anomalies in HVAC temperature data using multiple statistical and machine learning methods, with an interactive Streamlit app.

## Features

- Multiple anomaly detection algorithms: Z-score, Modified Z-score, IQR, Isolation Forest, Percentile  
- Supports datasets with up to 50,000 rows efficiently  
- Interactive visualizations highlighting anomalies  
- Export processed data with anomaly flags  

## Getting Started

### Prerequisites

- Python 3.9 or higher  
- See `requirements.txt` for package dependencies  

### Installation

1. Clone the repository  
git clone https://github.com/YourUsername/Predictive_Anamoly_HVAC.git
cd Predictive_Anamoly_HVAC

text

2. Create and activate virtual environment  
python -m venv predictive
source predictive/bin/activate # Windows: predictive\Scripts\activate

text

3. Install dependencies  
pip install -r requirements.txt

text

### Usage

Run the Streamlit app:  
streamlit run anamoly_till_5000.py

text

Upload a CSV with HVAC data columns and explore anomaly detection results.

## Sample Output

Add screenshots of your app here, for example:

<img width="987" height="542" alt="Screenshot 2025-09-01 152634" src="https://github.com/user-attachments/assets/29865edd-9124-45d5-a218-1ebe46aff67b" />
<img width="1887" height="566" alt="Screenshot 2025-09-01 152204" src="https://github.com/user-attachments/assets/c383cbff-a5a9-473c-8278-a45eb50cc7ca" />
<img width="1899" height="911" alt="Screenshot 2025-09-01 152138" src="https://github.com/user-attachments/assets/98720638-6b3c-4835-8687-2cdf1bf8b3b9" />
<img width="1905" height="914" alt="Screenshot 2025-09-01 152112" src="https://github.com/user-attachments/assets/f11067d8-a729-48d6-bdcc-14db94080562" />

![Anomaly Detection Output](images/anomaly_output.png)  
![Time Series with Anomalies](images/time_series.png)

## Project Structure

├── anamoly_for_till_50000.py
├── anamoly_till_5000.py
├── streamlit_hvac_interactive_app.py
├── data/
├── models/
├── predictive/ # virtual environment (ignored)
├── requirements.txt
├── README.md
└── images/ # screenshots


