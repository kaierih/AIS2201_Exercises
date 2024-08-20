import datetime as dt
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

def getTideObservations(N_days:int)->(np.ndarray, int, str, str):
    """
    Function to retrieve historical sea level data from kartverket for 
    Ålesund measuring station for the last 'N_days' days.

    Parameters:
    N_days - number of days to backtrack from current date

    Returns:
    measurements - Array of sea levels in cm relative to mean. 
                   The first measurement is taken at time 00:00 at the N_days days ago.
    meas_freq - number of measurements per day.

    start_date - String representing the date where log starts

    end_date - String representing the date where log ends
    """
    
    # Acquire date and time range in ISO8601 string format
    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=2)))
    log_end = today.replace(hour=0, minute=0, second=0, microsecond=0) - dt.timedelta(minutes=10)
    log_start = log_end - dt.timedelta(days=N_days-1, hours=23, minutes=50)

    to_time = log_end.isoformat()
    from_time = log_start.isoformat()
    
    # Generate API query parameters
    tidelevel_api = 'http://api.sehavniva.no/tideapi.php'
    params = {'lat':'62.469414',
          'lon': '6.151946',
          'fromtime': from_time,
          'totime': to_time,
          'datatype': 'obs',
          'refcode': 'msl',
          'lang':'en',
          'interval':'10',
          'dst':'0',
          'tide_request': 'locationdata'}

    # Get tide data from API
    response_API = requests.get(tidelevel_api, params=params)
    data_API=response_API.content    
    
    # Parse Data
    xmldata_root = ET.fromstring(data_API)
    waterlevel_measurements = []
    for x in xmldata_root.find('locationdata').iter('waterlevel'):
        waterlevel_measurements.append(x.attrib)
    df = pd.DataFrame(waterlevel_measurements)
    #df.style.set_caption(xmldata_root.find('locationdata').find('location').attrib['descr'])
    
    measurements = np.array(pd.to_numeric(df['value']))
    meas_freq = len(measurements)//N_days

    start_date = from_time.split("T")[0]
    end_date = to_time.split("T")[0]
    
    return measurements, meas_freq, start_date, end_date

def getTidePredictions(T:int):
    
    # Acquire date and time range in ISO8601 string format
    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=2)))
    today = today.replace(hour=0, minute=0, second=0, microsecond=0)
    pred_stop = today + dt.timedelta(days=T-1, hours=23, minutes=50)

    to_time = pred_stop.isoformat()
    from_time = today.isoformat()
    
    # Generate API query parameters
    tidelevel_api = 'http://api.sehavniva.no/tideapi.php'
    params = {'lat':'62.469414',
          'lon': '6.151946',
          'fromtime': from_time,
          'totime': to_time,
          'datatype': 'pre',
          'refcode': 'msl',
          'lang':'en',
          'interval':'10',
          'dst':'0',
          'tide_request': 'locationdata'}

    # Get tide data from API
    response_API = requests.get(tidelevel_api, params=params)
    data_API=response_API.content    
    
    # Parse Data
    xmldata_root = ET.fromstring(data_API)
    waterlevel_measurements = []
    for x in xmldata_root.find('locationdata').iter('waterlevel'):
        waterlevel_measurements.append(x.attrib)
    df = pd.DataFrame(waterlevel_measurements)
    #df.style.set_caption(xmldata_root.find('locationdata').find('location').attrib['descr'])
    measurements = np.array(pd.to_numeric(df['value']))
    meas_freq = len(measurements)//T
    
    return measurements, meas_freq, from_time.split("T")[0], to_time.split("T")[0]

def get_significant_fourier_coeffs(xt, K:int):
    Xf = np.fft.fft(xt)/len(xt)
    Xf_positive = Xf[0:len(xt)//2]
    k = np.sort(np.argpartition(abs(Xf_positive), -K)[-K:])
    c_k = Xf_positive[k]
    coeff_list = [{'k': index, 'c_k': Xf_positive[index] } for index in k]
    return coeff_list