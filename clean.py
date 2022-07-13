import glob
import os
import shutil


def clean_files(dir, file_type):
    for path in glob.glob(dir + f'/*.{file_type}'):
        os.remove(path)


def clean_dir(dir):
    for i in os.listdir(dir):
        path = os.path.join(dir,i)
        if os.path.isdir(path):
            shutil.rmtree(os.path.join(dir,i))


if __name__ == '__main__':
    remove_file_list = [
        ('aggrigator/config_f1_performance', 'pdf'),
        ('aggrigator/forplot', 'csv'),
        ('aggrigator/full', 'csv'),
        ('aggrigator/short', 'csv'),
        ('analysis/plots','pdf'),
        ('correlation_calc/plots', 'pdf'),
        ('correlation_calc', 'csv'),
        ('delta_calc/plot', 'pdf'),
    ]

    for i in remove_file_list:
        clean_files(i[0], i[1])

    remove_dir_list = [
        'analysis/plots',
        'analysis/tables'
    ]

    for i in remove_dir_list:
        clean_dir(i)
