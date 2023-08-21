import csv
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import sys
import argparse
import glob
import pandas as pd
import shutil
import argparse

# Add the command-line parsing here
parser = argparse.ArgumentParser(description="Process EQVIO data")
parser.add_argument('-d', '--data_directory', type=str, help="Path to the dataset directory")
parser.add_argument('-n', '--data_name', type=str, help="Name of the dataset")
parser.add_argument('-depthFlag', type=str, help='The depth flag')

# Parse the arguments
args = parser.parse_args()

# Convert depthFlag to boolean
args.depthFlag = args.depthFlag.lower() in ('true', '1', 't', 'y', 'yes')

# print('###processing {} data ######'.format(args.depthFlag == True))
print('##############processing {}############## '.format(args.data_name))

# Define the main directory containing all the datasets
if args.depthFlag:
    base_directory_path = '/media/fhc/Data/Dataset/depth_result/VCU/eqviodepth'
else:
    base_directory_path = '/media/fhc/Data/Dataset/depth_result/VCU/RGBeqvio'

# Modify ground truth file path
gt_directory_path = "/media/fhc/Data/Dataset/VCU/Groundtruth"

time_index_map = {
## handheld-lab-simple
'lab_simple1': 5645.179542390,
'lab_simple2': 5725.990606451,
'lab_simple3': 5800.002496175,
## handheld-lab-motion
'lab_motion1': 858.828367331,
'lab_motion2': 1179.124723715,
'lab_motion3': 620.74158628,
'lab_motion4': 795.027770684,
'lab_motion5': 55500.154868292,
'lab_motion6': 55631.844341273,
## handheld-lab-light
'lab_light1': 1430.872305079,
'lab_light2': 1569.199532825,
'lab_light3': 348.185657436,
'lab_light4': 25509.090028725,
'lab_light5': 1564.450962713,
'lab_light6': 1694.879961643,
## handheld-lab-dynamic
'lab_dynamic1': 1015.839053138,
'lab_dynamic2': 1382.506229709,
'lab_dynamic3': 1945.370618162,
'lab_dynamic4': 12815.547334773,
'lab_dynamic5': 13102.889270907,
## handheld-corridor
'corridor1':1429.843360791,
'corridor2':2567.387245158,
'corridor3':1120.039553804,
'corridor4':1388.710462629,
## handheld-hall
'hall1':2809.832627107,
'hall2':3416.940431843,
'hall3':226.740069808,
## robot-manual
'lab_manual1':193443.642084968,
'lab_manual2':193614.659717371,
'lab_manual3':193804.507317469,
## robot-bumper
'lab_bumper1':3348.721497606,
'lab_bumper2':3624.756204693,
'lab_bumper3':24305.238540249,
'lab_bumper4':24456.282658849,
'lab_bumper5':24861.813846655,
## robot-corridor-manual
'corridor_manual1': 768.813970368,
'corridor_manual2': 3604.818084375,
## robot-corridor-bumper
'corridor_bumper1': 4589.798765979,
'corridor_bumper2': 6756.646370621
}

def dataFromCSV(data_path):
    with open(data_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data = list(reader)
        # print(data[0])
        data = np.array(data[1:])  # Skip the first row (header)
        data[data == ''] = '0'
        data = data.astype(np.float64)
        data[:,0] = data[:,0]/1e9

    return data

def find_nearest(array, v):
    array  = np.asarray(array)
    idx = (np.abs(array-v)).argmin()
    return idx


def process_result(data_file, data_name):
    if data_name not in time_index_map:
        print("data_name: {} not in time_index_map".format(data_name))
        return

    ## input data
    data = dataFromCSV(data_file)

    ## find out the zero time for synchronization
    time_zero = time_index_map[data_name]
    start_idx = find_nearest(data[0,:], float(time_zero))

    ## extract the synchronized data
    data = data[start_idx:-1, 0:8]

    # convert to TUM: 'timestamp tx ty tz qx qy qz qw'
    # vins's output :w x y z'
    quatXYZ = np.copy(data[:,5:8])
    quatW = np.copy(data[:,4])
    data[:,4:7]= quatXYZ
    data[:,7]= quatW

    # set the start point as original point of the world coordinate system
    Tow = np.matrix(np.identity( 4 ))
    t = data[0,1:4]
    Tow[0:3,3] = t.reshape((3, 1))
    r = R.from_quat(data[0,4:8])

    Tow[0:3,0:3] = r.as_matrix()
    Tow = np.linalg.inv(Tow)

    for i in range(data.shape[0]):
        Twi = np.matrix(np.identity(4))
        Twi[0:3,3] = data[i,1:4].reshape((3,1))
        Twi[0:3,0:3] = R.from_quat(data[i,4:8]).as_matrix()

        Toi = Tow * Twi
        # Toi = Twi
        data[i,1:4] = Toi[0:3,3].reshape(3)
        data[i,4:8] = R.from_matrix(Toi[0:3,0:3]).as_quat()

    # output file
    output_file = data_file[:-4]+'_tum.csv'
    print('processed IMUstate file: {} '.format(output_file))
    # np.savetxt(output_file, data, delimiter=' ', fmt=('%10.9f', '%2.6f', '%2.6f', '%2.6f', '%2.6f', '%2.6f', '%2.6f', '%2.6f'))
    # at the end of process_result
    with open(output_file, 'w') as f:
        # Write the header
        f.write("#time px py pz qx qy qz qw\n")
        for line in data:
            timestamp = '{:.12f}'.format(line[0]*1e9)
            f.write("{} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f}\n".format(
                timestamp, line[1], line[2], line[3], line[4], line[5], line[6], line[7]))
    df = pd.DataFrame(data, columns=['time', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw'])

    return df

def process_dataset(data_file, data_name, depthFlag):
    # print(depthFlag==False)
    if data_name not in time_index_map:
        print("data_name: {} not in time_index_map".format(data_name))
        return
    # print(data_file,data_name)
    df1 = process_result(data_file, data_name)
    # use os.path.join to concatenate the directory path and file name
    file_path = os.path.join(args.data_directory, 'IMUState_tum.csv')
    # print('file paht {}'.format(file_path))
    # read the csv file from the provided path
    df = pd.read_csv(file_path,header=0, delimiter=' ')
    # print(df.columns)

    stamped_traj_estimate_file = os.path.join(os.path.dirname(data_file), 'stamped_traj_estimate.txt')
    with open(stamped_traj_estimate_file, 'w') as f:
        # Write the header
        f.write("#timestamp tx ty tz qx qy qz qw\n")
        # Write the data in scientific notation
        for index, row in df.iterrows():
            f.write("{} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f}\n".format(
                row['#time'], row['px'], row['py'], row['pz'], row['qx'], row['qy'], row['qz'], row['qw']))

    # Provided transformation matrix
    T = np.array([[0, 0, -1, 0], 
                    [0, 1, 0, 0], 
                    [1, 0, 0, 0], 
                    [0, 0, 0, 1]])

    T1 = np.array([[0,0,-1],
                    [0,1,0],
                    [1,0,0]])

    
    # Process the ground truth file to stamped_groundtruth.txt
    gt_file = os.path.join(gt_directory_path, data_name + '_gt.csv')  # Updated ground truth file path
    stamped_groundtruth_file = os.path.join(os.path.dirname(data_file), 'stamped_groundtruth.txt')
    df = pd.read_csv(gt_file, delimiter=" ", header=0) 
    df.columns = ['#time', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']

    # Apply the transformation to the position and orientation
    for index, row in df.iterrows():
        position = np.array([row['tx'], row['ty'], row['tz'], 1])
        orientation = R.from_quat([row['qx'], row['qy'], row['qz'], row['qw']])

        # Apply the transformation matrix to the position
        transformed_position = np.matmul(T, position)
        # Apply the transformation matrix to the orientation
        transformed_orientation = R.from_matrix(np.matmul(T1, orientation.as_matrix()))

        # Replace the original position and orientation with the transformed ones
        df.at[index, 'tx'] = transformed_position[0]
        df.at[index, 'ty'] = transformed_position[1]
        df.at[index, 'tz'] = transformed_position[2]
        df.at[index, 'qx'], df.at[index, 'qy'], df.at[index, 'qz'], df.at[index, 'qw'] = transformed_orientation.as_quat()

    with open(stamped_groundtruth_file, 'w') as fout:
        # Write the header
        fout.write('# time tx ty tz qx qy qz qw\n')

        # Iterate over the rows of the DataFrame
        for index, row in df.iterrows():
            # Format the output line
            line = "{:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f}\n".format(
                row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])

            # Write the line to the output file
            fout.write(line)


    # Write the eval config file
    eval_config_file = os.path.join(os.path.dirname(data_file), 'eval_cfg.yaml')
    with open(eval_config_file, 'w') as f:
        f.write('align_type: se3\n')
        f.write('align_num_frames: -1\n')
        f.write('start_time_sec: {:.9f}\n'.format(time_index_map[data_name]))

    # Copy the generated files to the corresponding directory in VCU_eqviodepth_Dataset
    if depthFlag:
        output_dir = os.path.join("/media/fhc/Data/Dataset/depth_result/VCU/eqviodepth/VCU_eqviodepth_" + data_name)
    else:
        output_dir = os.path.join("/media/fhc/Data/Dataset/depth_result/VCU/RGBeqvio/VCU_RGBeqvio_" + data_name)
    # print(output_dir)
    # print(data_name)
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # shutil.copyfile(imu_output_file, os.path.join(output_dir, 'imu_output.csv'))
    # shutil.copyfile(imu_output_tum_file, os.path.join(output_dir, 'imu_output_tum.csv'))
    shutil.copyfile(stamped_traj_estimate_file, os.path.join(output_dir, 'stamped_traj_estimate.txt'))
    shutil.copyfile(stamped_groundtruth_file, os.path.join(output_dir, 'stamped_groundtruth.txt'))
    shutil.copyfile(eval_config_file, os.path.join(output_dir, 'eval_cfg.yaml'))
    print("Copied files to {}".format(data_name))


# For each dataset in the base directory
for data_dir in os.listdir(base_directory_path):
    # Create the full path to the current dataset directory
    data_dir_path = os.path.join(base_directory_path, data_dir)

    # Check if it's a directory
    if os.path.isdir(data_dir_path):
        # For each csv file in the dataset directory
        for csv_file in glob.glob(os.path.join(data_dir_path, 'IMUState.csv')):
            # Extract the data sequence name from the directory name
            # Remove the 'VCU_eqviodepth_' prefix from the directory name to get the data_name
            if args.depthFlag:
                data_name = data_dir.replace('VCU_eqviodepth_', '')
                # Process the dataset
                process_dataset(csv_file, data_name, True)
            else:
                data_name = data_dir.replace('VCU_RGBeqvio_', '')
                process_dataset(csv_file, data_name, False)
            

if __name__ == "__main__":
    # If the required arguments are given, use them
    if args.data_directory and args.data_name:
        # Find the IMUState.csv file in the given directory
        for csv_file in glob.glob(os.path.join(args.data_directory, 'IMUState.csv')):
            # Process the dataset
            process_dataset(csv_file, args.data_name, args.depthFlag)
    else:
        # Otherwise, process all datasets
        print("wrong folder or dataset name")

print('####################### processd imu result #######################')

#python analyze_result.py -d /home/fhc/ANU/eqvio/EQVIO_depthoutput_2023-07-30_13:01:42/ -n hall1 -depthFlag 0