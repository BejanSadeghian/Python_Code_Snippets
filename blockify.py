########################################
## Author: Bejan Lee Sadeghian
## Purpose: Break up larger datasets into managable pieces but ensure that all items of a certain group are in only one block (none broken up into two blocks)
## Warning: Input data MUST be ordered by the column that we want to make sure its mutually excluded from >1 block
########################################

import pandas as pd
import os

n = 500000 #approximate number of rows for each block
root_directory = r'C:\Users\\'
file_url = r'example_input.csv'
output_directory = r'Python Scripts\Temp\\'
column_seperator = 'column_name_to_seperate_on'

## BREAK LARGE DATASETS INTO BLOCKS
def blockify(root_directory, file_url, output_directory, column_seperator, chunk_size = 500000):
    """ DATA MUST BE SORTED BY COLUMN SEPERATOR and ALL data in the TEMP Folder will be deleted """
    folder = root_directory + output_directory
    user_in = raw_input('Do you want to delete the original block data in the folder "' + folder + '"? (type yes): ')
    if user_in.lower() == 'yes':
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path) and the_file[:11] == 'data_block_':
                    os.unlink(file_path)
            except Exception as e:
                print(e)
    #Begin creating the blocks
    last_id = ''
    data = pd.read_csv(root_directory + file_url, iterator = True, chunksize = n)
    for i, df in enumerate(data):
        state = i
        if last_id == '':
            last_id = df.iloc[df.shape[0]-1][column_seperator]
            block_out = df.ix[df.ix[:,column_seperator] != last_id,:]
            block_out.to_csv(root_directory + output_directory + 'data_block_' + str(i) + '.csv', index=False)
            block_out = df.ix[df.ix[:,column_seperator] == last_id,:]
        else:
            last_id = df.iloc[df.shape[0]-1][column_seperator]
            block_out = block_out.append(df.ix[df.ix[:,column_seperator] != last_id,:], ignore_index=True)
            block_out.to_csv(root_directory + output_directory + 'data_block_' + str(i) + '.csv', index=False)
            block_out = df.ix[df.ix[:,column_seperator] == last_id,:]
    # Add back the last patient to the last data block
    temp_read = pd.read_csv(root_directory + output_directory + 'data_block_' + str(state) + '.csv')
    temp_read = temp_read.append(block_out)
    temp_read.to_csv(root_directory + output_directory + 'data_block_' + str(state) + '.csv', index=False)

## Execution
#blockify(root_directory, file_url, output_directory, column_seperator, n)
            