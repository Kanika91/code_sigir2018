#coding: utf8

import os, shutil
from termcolor import colored
import ConfigParser as cp

class Logging():
    def __init__(self, filename, debug_flag):
        self.filename = filename
        self.debug_flag = debug_flag
    
    def record(self, str_log):
        debug_flag = self.debug_flag
        if debug_flag == 1:
            filename = self.filename
            print(str_log)
            with open(filename, 'a') as f:
                f.write("%s\r\n" % str_log)
                f.flush()
        elif debug_flag == 0:
            print(str_log)

    def record_c(self, str_log, color):
        debug_flag = self.debug_flag
        if debug_flag == 1:
            filename = self.filename
            print(colored(str_log, color))
            with open(filename, 'a') as f:
                f.write("%s\r\n" % str_log)
                f.flush()

    def debug_function(self):
        config_filename = self.config_filename
        debug_flag = self.debug_flag
        if debug_flag == 1:
            config = cp.ConfigParser()
            config.read(config_filename)
            counter = config.get('Debug', 'counter')
            src_dir = config.get('Debug', 'src_dir')
            dst_dir = config.get('Debug', 'dst_dir')
            dst_dir = "%s_%s_" % (dst_dir, counter)
            src_filename1 = config.get('Debug', 'src_filename1')
            src_filename2 = config.get('Debug', 'src_filename2')
            src_path1 = "%s%s" % (src_dir, src_filename1)
            src_path2 = "%s%s" % (src_dir, src_filename2)
            dst_path1 = "%s_%d_%s" % (dst_dir, counter, src_filename1)
            dst_path2 = "%s_%d_%s" % (dst_dir, counter, src_filename2)
    
            ## copy ini, code.py to destination
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            shutil.copyfile(src_path1, dst_path1)
            shutil.copyfile(src_path2, dst_path2)

            ## update ini counter
            counter += 1
            config.set('Debug', 'counter', '%d'%counter)
            config.write(open(config_filename, 'w'))

