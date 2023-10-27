import numpy as np
import cmath
import csv
import os


def process_antennas(num_antennas, num_subcarriers, folder):
    folder_samples = folder + '/samples/'
    antenna_pos = np.load(folder + '/antenna_positions.npy')
    user_pos = np.load(folder + '/user_positions.npy')

    # Indices of the selected antennas
    if num_antennas == 64:
        selected_antennas = [x for x in range(64)]
    elif num_antennas == 32:
        if folder == "DIS_lab_LoS":
            selected_antennas = [2, 3, 4, 5, 10, 11, 12, 13, 18, 19, 20, 21,
                                 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44,
                                 45, 50, 51, 52, 53, 58, 59, 60, 61]
        elif folder == "URA_lab_LoS":
            selected_antennas = [10, 11, 12, 13, 17, 18, 19, 20, 21, 22,
                                 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 37, 38,
                                 41, 42, 43, 44, 45, 46, 50, 51, 52, 53]
        elif folder == "ULA_lab_LoS":
            selected_antennas = [x + 16 for x in range(32)]
    elif num_antennas == 16:
        if folder == "DIS_lab_LoS":
            selected_antennas = [3, 4, 11, 12, 19, 20, 27, 28,
                                 35, 36, 43, 44, 51, 52, 59, 60]
        elif folder == "URA_lab_LoS":
            selected_antennas = [18, 19, 20, 21, 26, 27, 28, 29,
                                 34, 35, 36, 37, 42, 43, 44, 45]
        elif folder == "ULA_lab_LoS":
            selected_antennas = [x + 24 for x in range(16)]
    elif num_antennas == 8:
        if folder == "DIS_lab_LoS":
            selected_antennas = [3 + 8*x for x in range(8)]
        elif folder == "URA_lab_LoS":
            selected_antennas = [26, 27, 28, 29, 34, 35, 36, 37]
        elif folder == "ULA_lab_LoS":
            selected_antennas = [x + 28 for x in range(8)]
    rows = []
    
    user_coordinates = list(user_pos)

    for filename in os.listdir(folder_samples):
        file_path = os.path.join(folder_samples, filename)
        measurements = np.load(file_path)


        # Create matrices to store the measurements
        modulo = np.zeros((num_antennas, num_subcarriers))
        angulo = np.zeros((num_antennas, num_subcarriers))

        # Fill the matrices with the measurements
        for antena_idx, antena_measurement in enumerate(selected_antennas):
            for portadora_idx in range(num_subcarriers):
                portadora_measurement = measurements[antena_measurement][portadora_idx]
                modulo[antena_idx][portadora_idx] = cmath.polar(portadora_measurement)[0]
                angulo[antena_idx][portadora_idx] = cmath.polar(portadora_measurement)[1]


        pos_idx = int(filename.split('_')[2][:-4])
        user_coordinate = user_coordinates[pos_idx]

        row = []
        
        
        # Append the measurements to the row
        for antena_idx in range(num_antennas):
            for portadora_idx in range(num_subcarriers):
                row.append(modulo[antena_idx][portadora_idx])
                row.append(angulo[antena_idx][portadora_idx])
                
        row.extend(user_coordinate[:2])
        rows.append(row)

        
    # Create csv
    headers = []
    for antena_idx in range(num_antennas):
        for portadora_idx in range(num_subcarriers):
            headers.append(f'Antenna{antena_idx + 1}Subcarrier{portadora_idx + 1}Module')
            headers.append(f'Antenna{antena_idx + 1}Subcarrier{portadora_idx + 1}Angle')
    headers.extend(['PositionX', 'PositionY'])

    with open(f'{folder}_{num_antennas}_{num_subcarriers}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)

# ? AVAILABLE FOLDERS: 'DIS_lab_LoS', 'ULA_lab_LoS', 'URA_lab_LoS'
# To generate the .csv files, the data folders must be in the same directory as this file

process_antennas(num_antennas=8, num_subcarriers=100, folder='ULA_lab_LoS')
process_antennas(num_antennas=16, num_subcarriers=100, folder='ULA_lab_LoS')
process_antennas(num_antennas=32, num_subcarriers=100, folder='ULA_lab_LoS')
process_antennas(num_antennas=64, num_subcarriers=100, folder='ULA_lab_LoS')

process_antennas(num_antennas=8, num_subcarriers=100, folder='URA_lab_LoS')
process_antennas(num_antennas=16, num_subcarriers=100, folder='URA_lab_LoS')
process_antennas(num_antennas=32, num_subcarriers=100, folder='URA_lab_LoS')
process_antennas(num_antennas=64, num_subcarriers=100, folder='URA_lab_LoS')

"""process_antennas(num_antennas=8, num_subcarriers=100, folder='DIS_lab_LoS')
process_antennas(num_antennas=16, num_subcarriers=100, folder='DIS_lab_LoS')
process_antennas(num_antennas=32, num_subcarriers=100, folder='DIS_lab_LoS')
process_antennas(num_antennas=64, num_subcarriers=100, folder='DIS_lab_LoS')"""

