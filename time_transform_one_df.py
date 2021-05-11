import os
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import pytz

'''обов'язково лишити свій час і день тижня
додати коди лєвацька моя ф-я внизу
destinaton to destination'''


def get_icao():
    airports = pd.read_csv('unique/airports_relevant_z.csv')
    flights_data = pd.read_csv('unique/all_flights_data.csv')

    origin_i = [airports[airports.iata_code == ia].gps_code for ia in flights_data.origin.to_list()]
    origin_i_final = [i.values[0] for i in origin_i]
    flights_data['origin_icao'] = origin_i_final

    destination_i = [airports[airports.iata_code == ia].gps_code for ia in flights_data.destination.to_list()]
    destination_i_final = [i.values[0] for i in destination_i]
    flights_data['destination_icao'] = destination_i_final

    flights_data.to_csv('unique/all_flights_data_codes.csv', index=False)


def weather_countries_to_airports():
    for country_weather_path in os.listdir('data_weather'):
        country_weather = pd.read_csv("data_weather/" + country_weather_path)
        for airport_icao in country_weather.station.unique():
            airport_weather = country_weather[country_weather.station == airport_icao]
            airport_weather.to_csv('data_weather_airports/' + airport_icao + '.csv', index=False)


def clean_unnecessary_weather():
    for country_weather_path in os.listdir('data_weather_all'):
        print('\n')
        print(country_weather_path)
        country_weather = pd.read_csv("data_weather_all/" + country_weather_path)
        country_weather.drop(
            ["p01i", "skyc1", "skyc2", "skyc3", "skyc4", "skyl1", "skyl2", "skyl3", "skyl4", "mslp", "gust", "wxcodes"],
            inplace=True, axis=1)
        country_weather.drop(
            ["ice_accretion_1hr", "metar", "ice_accretion_3hr", "ice_accretion_6hr", "peak_wind_gust", "peak_wind_drct",
             "peak_wind_time"], inplace=True, axis=1)
        country_weather = country_weather.replace('M', np.NaN)
        country_weather["valid"] = country_weather["valid"].astype(np.datetime64)
        country_weather.to_csv('data_weather/' + country_weather_path, index=False)


def list_of_airports_countries():
    airports_occured = []

    for model_flights_path in os.listdir('data_csv'):
        model_flights = pd.read_csv('data_csv/' + model_flights_path)
        airports_occured = np.concatenate((airports_occured, model_flights.Origin.unique()))
        airports_occured = np.unique(airports_occured)

    airports_occured = pd.DataFrame(airports_occured, columns=['iata_code'])
    print(len(airports_occured))
    airports = pd.read_csv('unique/airports_relevant_z.csv')
    print(len(airports))
    merged = pd.merge(airports, airports_occured, how='right', on='iata_code')
    print(len(merged))

    for country in merged.iso_country.unique():
        print('country code', country)
        print(merged[merged.iso_country == country].gps_code)
        print(merged[merged.iso_country == country].timezone)
        print('\n' * 2)


def change_timezone(initial_datetime, initial_timezone):
    local = pytz.timezone(initial_timezone)
    local_dt = local.localize(initial_datetime, is_dst=None)
    utc_dt = local_dt.astimezone(pytz.utc)
    return utc_dt


def get_times_delays_utc(flight_raw):
    airports = pd.read_csv("unique/airports_relevant_z.csv")
    # print(flight_raw["Origin"].values[0])
    timezone_origin = airports[airports.iata_code == flight_raw["Origin"].values[0]].timezone.values[0]
    timezone_destination = airports[airports.iata_code == flight_raw["Destination"].values[0]].timezone.values[0]

    try:
        this_flight_STD = flight_raw["Date of flight"].values[0] + ' ' + flight_raw["STD"].values[0].replace("\t",
                                                                                                             '').strip()
        this_flight_STD = datetime.strptime(this_flight_STD, "%d %b %Y %I:%M %p")
        this_flight_STD_utc = change_timezone(this_flight_STD, timezone_origin)
    except:
        this_flight_STD, this_flight_STD_utc = np.NaN, np.NaN

    try:
        this_flight_ATD = flight_raw["Date of flight"].values[0] + ' ' + flight_raw["ATD"].values[0].replace("\t",
                                                                                                             '').strip()
        this_flight_ATD = datetime.strptime(this_flight_ATD, "%d %b %Y %I:%M %p")
        this_flight_ATD_utc = change_timezone(this_flight_ATD, timezone_origin)
    except:
        this_flight_ATD, this_flight_ATD_utc = np.NaN, np.NaN

    try:
        this_flight_STA = flight_raw["Date of flight"].values[0] + ' ' + flight_raw["STA"].values[0].replace("\t",
                                                                                                             '').strip()
        this_flight_STA = datetime.strptime(this_flight_STA, "%d %b %Y %I:%M %p")
        this_flight_STA_utc = change_timezone(this_flight_STA, timezone_destination)
    except:
        this_flight_STA, this_flight_STA_utc = np.NaN, np.NaN

    try:
        this_flight_ATA = flight_raw["Date of flight"].values[0] + ' ' + flight_raw["ATA"].values[0].replace("Landed",
                                                                                                             '').replace(
            "\t", '').strip()
        this_flight_ATA = datetime.strptime(this_flight_ATA, "%d %b %Y %I:%M %p")
        this_flight_ATA_utc = change_timezone(this_flight_ATA, timezone_destination)
    except:
        this_flight_ATA_utc, this_flight_ATA = np.NaN, np.NaN

    '''print(this_flight_STD)
    print(this_flight_ATD)
    print(this_flight_STA)
    print(this_flight_ATA)
    print(this_flight_STD_utc)
    print(this_flight_ATD_utc)
    print(this_flight_STA_utc)
    print(this_flight_ATA_utc)'''

    no_delay = timedelta(hours=0, minutes=0, seconds=0)

    try:
        if this_flight_ATD > this_flight_STD:
            this_flight_D_delay = this_flight_ATD - this_flight_STD
        else:
            this_flight_D_delay = no_delay
    except:
        this_flight_D_delay = None

    try:
        if this_flight_ATA > this_flight_STA:
            this_flight_A_delay = this_flight_ATA - this_flight_STA
        else:
            this_flight_A_delay = no_delay
    except:
        this_flight_A_delay = None

    return this_flight_STD_utc, this_flight_ATD_utc, this_flight_STA_utc, this_flight_ATA_utc, this_flight_D_delay, this_flight_A_delay, this_flight_STD, this_flight_ATD, this_flight_STA, this_flight_ATA


def all_flights_to_one_df():
    '''видалити аеропорти яких немає в погоді'''
    airports_no_weather = ['TEV', 'PLV', 'NWI', 'HEM', 'DHF', 'TNL', 'ZTR']
    models = pd.read_csv("unique/models_info.csv")

    all_flights_df = pd.DataFrame()
    l = 0

    for model_flights_path in os.listdir('data_csv'):
        try:
            model_flights = pd.read_csv('data_csv/' + model_flights_path)
            print('Reading', model_flights_path)
            model_flights = model_flights[~model_flights.Origin.isin(airports_no_weather)]
            model_flights = model_flights[~model_flights.Destination.isin(airports_no_weather)]
            model_flights = model_flights.replace('-', np.NaN).replace('Unknown', np.NaN).replace('Ã¢â‚¬â€',
                                                                                                  np.NaN).replace('â€”',
                                                                                                                  np.NaN)
            model_flights.reset_index(drop=True, inplace=True)
            origin_f = model_flights.Origin.to_list()
            destination_f = model_flights.Destination.to_list()
            model_f = model_flights.Model.to_list()
            code_f = [models.loc[models["Registration number"] == model_f[0], ['Type code']].values[0][0]] * len(
                origin_f)
            operator_f = [models.loc[models["Registration number"] == model_f[0], ['Operator']].values[0][0]] * len(
                origin_f)
            production_year = models.loc[models["Registration number"] == model_f[0], ['Production']].values[0][0][-4:]
            production_f = [production_year] * len(origin_f)
            flight_number_f = model_flights["Flight number"].to_list()

            STDs_utc, ATDs_utc, STAs_utc, ATAs_utc, DDs, ADs, STDs, ATDs, STAs, ATAs = [], [], [], [], [], [], [], [], [], []

            for i in range(len(model_flights)):
                flight_raw = model_flights.iloc[[i]]
                STD_utc, ATD_utc, STA_utc, ATA_utc, DD, AD, STD, ATD, STA, ATA = get_times_delays_utc(flight_raw)
                STDs_utc.append(STD_utc)
                ATDs_utc.append(ATD_utc)
                STAs_utc.append(STA_utc)
                ATAs_utc.append(ATA_utc)
                STDs.append(STD)
                ATDs.append(ATD)
                STAs.append(STA)
                ATAs.append(ATA)
                DDs.append(DD)
                ADs.append(AD)

            STD_f_utc, ATD_f_utc, STA_f_utc, ATA_f_utc, DD_f, AD_f, STD_f, ATD_f, STA_f, ATA_f = STDs_utc, ATDs_utc, STAs_utc, ATAs_utc, DDs, ADs, STDs, ATDs, STAs, ATAs
            delayed_departure_f = [d > timedelta(seconds=1) if d is not None else np.NaN for d in DD_f]
            delayed_arrival_f = [d > timedelta(seconds=1) if d is not None else np.NaN for d in AD_f]

            this_flights_df = pd.DataFrame(
                {'origin': origin_f,
                 'destination': destination_f,
                 'model': model_f,
                 'code': code_f,
                 'operator': operator_f,
                 'production': production_f,
                 'flight_number': flight_number_f,
                 'STD_utc': STD_f_utc,
                 'ATD_utc': ATD_f_utc,
                 'STA_utc': STA_f_utc,
                 'ATA_utc': ATA_f_utc,
                 'STD': STD_f,
                 'ATD': ATD_f,
                 'STA': STA_f,
                 'ATA': ATA_f,
                 'departure_delay': DD_f,
                 'arrival_delay': AD_f,
                 'is_delayed_arrival': delayed_arrival_f,
                 'is_delayed_departure': delayed_departure_f,
                 }
            )
            all_flights_df = all_flights_df.append(this_flights_df, ignore_index=True)
            all_flights_df.to_csv("unique/all_flights_data_new.csv", index=False)
            l = l + len(model_flights)
            print('Done with ', model_flights_path, ' expected len', str(l), '\n\n')
        except:
            print("Failed with", model_flights_path)
            continue

    # all_flights_df.to_csv("unique/all_flights_data.csv", index=False)


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


def append_weather_data():
    all_flights = pd.read_csv('unique/all_flights_data_codes.csv')
    all_flights['valid_a'], all_flights['lon_a'], all_flights['lat_a'], all_flights['elevation_a'], all_flights['tmpf_a'], \
    all_flights['dwpf_a'], all_flights[
        'relh_a'] = np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
    all_flights['drct_a'], all_flights['sknt_a'], all_flights['p01i_a'], all_flights['alti_a'], all_flights['mslp_a'], \
    all_flights['vsby_a'], all_flights['gust_a'], all_flights[
        'skyc1_a'] = np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
    all_flights['skyc2_a'], all_flights['skyc3_a'], all_flights['skyc4_a'], all_flights['skyl1_a'], all_flights['skyl2_a'], \
    all_flights['skyl3_a'], all_flights['skyl4_a'], all_flights[
        'wxcodes_a'] = np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
    all_flights['ice_accretion_1hr_a'], all_flights['ice_accretion_3hr_a'], all_flights['ice_accretion_6hr_a'], all_flights[
        'peak_wind_gust_a'], all_flights[
        'peak_wind_drct_a'], all_flights['peak_wind_time_a'], all_flights['feel_a'], all_flights[
        'metar_a'] = np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN

    for i in range(len(all_flights)):
        #тут дестінейшн і погодка для нього
        try:
            flight = all_flights.loc[[i]]
            weather_o = pd.read_csv('data_weather/' + flight.destination_icao.values[0] + '.csv')
            # print(weather_o.info())
            weather_o.dropna(inplace=True)
            weather_o["valid"] = weather_o["valid"].astype(np.datetime64)
            flight.STA_utc = flight.STA_utc.astype(np.datetime64)
            # print(weather_o.info())
            scheduled_arrival = flight.STA_utc.values[0]
            # print(type(scheduled_departure))
            # print(type(weather_o.valid))
            print(i)
            nearest_timestamp = nearest(weather_o.valid, scheduled_arrival)
            weather_of_interest = weather_o[weather_o.valid == nearest_timestamp]

            all_flights.loc[i, 'lon_a'] = weather_of_interest['lon'].values[0]
            all_flights.loc[i, 'valid_a'] = weather_of_interest['valid'].values[0]
            all_flights.loc[i, 'lat_a'] = weather_of_interest['lat'].values[0]
            all_flights.loc[i, 'elevation_a'] = weather_of_interest['elevation'].values[0]
            all_flights.loc[i, 'tmpf_a'] = weather_of_interest['tmpf'].values[0]
            all_flights.loc[i, 'dwpf_a'] = weather_of_interest['dwpf'].values[0]
            all_flights.loc[i, 'relh_a'] = weather_of_interest['relh'].values[0]
            all_flights.loc[i, 'drct_a'] = weather_of_interest['drct'].values[0]
            all_flights.loc[i, 'sknt_a'] = weather_of_interest['sknt'].values[0]
            all_flights.loc[i, 'p01i_a'] = weather_of_interest['p01i'].values[0]
            all_flights.loc[i, 'alti_a'] = weather_of_interest['alti'].values[0]
            all_flights.loc[i, 'mslp_a'] = weather_of_interest['mslp'].values[0]
            all_flights.loc[i, 'vsby_a'] = weather_of_interest['vsby'].values[0]
            all_flights.loc[i, 'gust_a'] = weather_of_interest['gust'].values[0]
            all_flights.loc[i, 'skyc1_a'] = weather_of_interest['skyc1'].values[0]
            all_flights.loc[i, 'skyc2_a'] = weather_of_interest['skyc2'].values[0]
            all_flights.loc[i, 'skyc3_a'] = weather_of_interest['skyc3'].values[0]
            all_flights.loc[i, 'skyc4_a'] = weather_of_interest['skyc4'].values[0]
            all_flights.loc[i, 'skyl1_a'] = weather_of_interest['skyl1'].values[0]
            all_flights.loc[i, 'skyl2_a'] = weather_of_interest['skyl2'].values[0]
            all_flights.loc[i, 'skyl3_a'] = weather_of_interest['skyl3'].values[0]
            all_flights.loc[i, 'skyl4_a'] = weather_of_interest['skyl4'].values[0]
            all_flights.loc[i, 'wxcodes_a'] = weather_of_interest['wxcodes'].values[0]
            all_flights.loc[i, 'ice_accretion_1hr_a'] = weather_of_interest['ice_accretion_1hr'].values[0]
            all_flights.loc[i, 'ice_accretion_3hr_a'] = weather_of_interest['ice_accretion_3hr'].values[0]
            all_flights.loc[i, 'ice_accretion_6hr_a'] = weather_of_interest['ice_accretion_6hr'].values[0]
            all_flights.loc[i, 'peak_wind_gust_a'] = weather_of_interest['peak_wind_gust'].values[0]
            all_flights.loc[i, 'peak_wind_drct_a'] = weather_of_interest['peak_wind_drct'].values[0]
            all_flights.loc[i, 'peak_wind_time_a'] = weather_of_interest['peak_wind_time'].values[0]
            all_flights.loc[i, 'feel_a'] = weather_of_interest['feel'].values[0]
            all_flights.loc[i, 'metar_a'] = weather_of_interest['metar'].values[0]
        except:
            print('PROBLEM WITH INDEEX', i)

        if i % 100 == 0:
            all_flights.to_csv("unique/all_flights_data_weather_new_dest_weather.csv", index=False)

    all_flights.to_csv("unique/all_flights_data_weather_new_dest_weather.csv", index=False)


if __name__ == "__main__":
    #append_weather_data()
    print("happiness is all around")
