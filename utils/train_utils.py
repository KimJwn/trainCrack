import os
import logging


def create_save_dir(name):
    default_path = "result"
    try:
        if not os.path.exists(default_path):
            os.makedirs(default_path)
        
        if not os.path.exists(default_path + '/' + name):
            os.makedirs(default_path + '/' + name)

    except:
        print("Error : Failed to create the directory")

    path = default_path + '/' + name +'/'
    
    return path

def get_log(path, mode='train'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if mode == 'train':
        file_handler = logging.FileHandler(path+'train.log')
    else:
        file_handler = logging.FileHandler(path + 'eval.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
