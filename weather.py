import os
import time
from datetime import datetime
import pandas as pd
from urllib.request import urlopen
import numpy as np


def download_data(uri):
    """Fetch the data from the IEM
    The IEM download service has some protections in place to keep the number
    of inbound requests in check.  This function implements an exponential
    backoff to keep individual downloads from erroring.
    Args:
      uri (string): URL to fetch
    Returns:
      string data
    """
    attempt = 0
    while attempt < 6:
        try:
            data = urlopen(uri, timeout=300).read().decode("utf-8")
            if data is not None and not data.startswith("ERROR"):
                return data
        except Exception as exp:
            print("download_data(%s) failed with %s" % (uri, exp))
            time.sleep(5)
        attempt += 1

    print("Exhausted attempts to download, returning empty data")
    return ""


def load_weather_stations(stations):
    startts = datetime(2018, 3, 1)
    endts = datetime(2021, 4, 15)

    SERVICE = "http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
    service = SERVICE + "data=all&tz=Etc/UTC&format=comma&latlon=yes&elev=yes&"

    service += startts.strftime("year1=%Y&month1=%m&day1=%d&")
    service += endts.strftime("year2=%Y&month2=%m&day2=%d&")

    for station in stations:
        try:
            uri = "%s&station=%s" % (service, station)
            print(uri)
            print("Downloading: %s" % (station,))
            data = download_data(uri)
            data = data.split('\n')[5:]
            for i in range(len(data)):
              data[i] = data[i].split(',')
            df_data = pd.DataFrame(data = data[1:], columns=data[0])
            df_data.to_csv('data_weather_auto/' + station + '.csv', index=False)
        except:
            print(station)
            continue


def get_unique_airports():
    airports_occured = []

    for model_flights_path in os.listdir('data_csv'):
        model_flights = pd.read_csv('data_csv/' + model_flights_path)
        airports_occured = np.concatenate((airports_occured, model_flights.Origin.unique()))
        airports_occured = np.concatenate((airports_occured, model_flights.Destination.unique()))
        airports_occured = np.unique(airports_occured)

    return airports_occured


def transform_codes(stations):
    airports = pd.read_csv('unique/airports_relevant_z.csv')
    for i in range(len(stations)):
        try:
            stations[i] = airports.loc[airports.iata_code == stations[i], 'gps_code'].values[0]
        except:
            stations[i] = 'not found'
    stations[191] = 'LHPR'
    stations[217] = 'EDDB'
    stations[231] = 'UKLT'
    return stations



if __name__ == "__main__":
    print("this part is done")