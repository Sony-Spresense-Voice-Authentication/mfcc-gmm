import os
import logging

def check_folder(folder_name):
    logging.info(f'Checking if {folder_name} directory exists')
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        logging.info(f'Creating {folder_name} directory')