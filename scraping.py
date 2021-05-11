import os
from bs4 import BeautifulSoup
import pandas as pd


def get_aircraft_data_html(path):
    soup = BeautifulSoup(open(path), features="html.parser")

    main_info = soup.find_all("div", class_="row h-30 p-l-20 p-t-5")

    registration_number = soup.title.string.split()[0]
    print('regisration number of this aircraft {}'.format(registration_number))

    if main_info[0].label.string == 'AIRCRAFT':
        aircraft_full = main_info[0].span.string.strip()
        print('full model is {}'.format(aircraft_full))

    if main_info[1].label.string == 'AIRLINE':
        airline = main_info[1].span.a.string.strip().replace('\t', '')
        print('airline operating this aircraft {}'.format(airline))

    if main_info[2].label.string == 'OPERATOR':
        operator = main_info[2].span.string.strip()
        print('operator {}'.format(operator))

    if main_info[3].label.string == 'TYPE CODE':
        type_code = main_info[3].span.string.strip()
        print('type code {}'.format(type_code))

    if main_info[4].label.string == 'Code':
        code = main_info[4].span.string.strip()
        print('code {}'.format(code))

    if main_info[6].label.string == 'MODE S':
        mode_s = main_info[6].span.string.strip()
        print('mode s {}'.format(mode_s))

    if main_info[7].label.string == 'SERIAL NUMBER (MSN)':
        serial_number = main_info[7].span.string.strip()
        print('serial number {}'.format(serial_number))

    if "AGE" in main_info[8].label.string:
        age = main_info[8].span.string.strip()
        print('age {}'.format(age))
        production = main_info[8].label.string.replace('AGE', '').replace('(', '').replace(')', '').strip()
        print('produced {}'.format(production))

    #тут вернути словником і досягати по назві а не по номери в списку
    return [registration_number, aircraft_full, airline, operator, type_code, code, mode_s, serial_number, age, production]

def reading_flights_to_df(path):

      aircraft_name = path.replace('.html', '').split('/')[-1]
      soup = BeautifulSoup(open(path), features="html.parser")
      flights_info = soup.find_all("tr", class_="data-row")  # don`t have to use spaces at all
      dates = []
      origins = []
      destinations = []
      flight_numbers = []
      flight_times = []
      STDs = []
      ATDs = []
      STAs = []
      ATAs = []

      for one_flight in flights_info:
          origin, destination, STD, ATD, ATA, STA, flight_number, flight_time, date = '', '', '', '', '', '', '', '', ''
          try:
              part1 = one_flight.find("div", class_="col-xs-8")
              times = part1.findAll("div", class_ = "col-xs-4")
              airports = part1.findAll("a")
              origin = airports[0].get_text().replace('(', '').replace(')','')
              destination = airports[1].get_text().replace('(', '').replace(')','')

              if "STD" in times[0].get_text().replace('\n', ''):
                STD = times[0].get_text().replace('\n', '')[3:]

              if "ATD" in times[1].get_text().replace('\n', ''):
                ATD = times[1].get_text().replace('\n', '')[3:]

              if "STA" in times[2].get_text().replace('\n', ''):
                STA = times[2].get_text().replace('\n', '')[3:]

              part2 = one_flight.findAll("td", class_ = "hidden-xs hidden-sm")
              date = part2[0].get_text()
              flight_number = part2[3].get_text()
              flight_time = part2[4].get_text()
              ATA = part2[9].get_text()
              if origin == '' or destination == '' or STD == '' or ATA == '' or ATD == '' or STA == '' or date == '' or flight_number == '' or flight_time == '':
                raise Exception
          except Exception:
            continue
          except:
            continue
          origins.append(origin)
          destinations.append(destination)
          STDs.append(STD)
          ATDs.append(ATD)
          STAs.append(STA)
          dates.append(date)
          flight_numbers.append(flight_number)
          flight_times.append(flight_time)
          ATAs.append(ATA)

      models = [aircraft_name] * len(dates)

      this_aircraft_flights_df = pd.DataFrame(
          {'Date of flight': dates,
           'Origin': origins,
           'Destination': destinations,
           'Flight number': flight_numbers,
           'Flight_times': flight_times,
           'Model': models,
           'STD': STDs,
           'ATD': ATDs,
           'STA': STAs,
           'ATA': ATAs
           })

      return this_aircraft_flights_df

def html_to_csv_xsls_flights(path):
    df_of_flights = reading_flights_to_df(path)
    aircraft_name = path.replace('.html', '').replace('-', '_')
    aircraft_name = aircraft_name.split('/')[-1]
    df_of_flights.to_excel('data_xlsx/' + str(aircraft_name) + '.xlsx', index = False)
    df_of_flights.to_csv('data_csv/' + str(aircraft_name) + '.csv', index = False)

registration_numbers, full_models, airlines, operators, type_codes, codes, modes_s, serial_numbers, ages, productions = [], [], [], [], [], [], [], [], [], []

for folder in os.listdir('data_f24'):
    path = 'data_f24/' + str(folder)
    for filename in os.listdir(path):
        if '.html' in filename:
            print(filename)
            model_details = get_aircraft_data_html(path + '/' + filename)
            registration_numbers.append(model_details[0])
            full_models.append(model_details[1])
            airlines.append(model_details[2])
            operators.append(model_details[3])
            type_codes.append(model_details[4])
            codes.append(model_details[5])
            modes_s.append(model_details[6])
            serial_numbers.append(model_details[7])
            ages.append(model_details[8])
            productions.append(model_details[9])
            #html_to_csv_xsls_flights(path + '/' + filename)

models_df = pd.DataFrame({
    'Registration number': registration_numbers,
    'Full_model': full_models,
    'Airline': airlines,
    'Operator': operators,
    'Type code': type_codes,
    'Code': codes,
    'Mode s': modes_s,
    'Serial number': serial_numbers,
    'Age': ages,
    'Production': productions
})

models_df.to_excel('data_xlsx/models_info.xlsx', index=False)
models_df.to_csv('data_csv/models_info.csv', index=False)