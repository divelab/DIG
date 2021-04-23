"""
FileName: logger.py
Description: Logger definition
Time: 2020/7/30 9:10
Project: GNN_benchmark
Author: Shurui Gui
"""


from cilog import create_logger, json_mail_setting
import os
from definitions import ROOT_DIR
import json
from benchmark.args import GeneralArgs

general_args = GeneralArgs().parse_args(known_only=True)

with open(os.path.join(ROOT_DIR, 'config', 'mail_setting.json')) as f:
    mail_setting = json.load(f)
    mail_setting = json_mail_setting(mail_setting)

create_logger(name='GNN_log',
              file=os.path.join(ROOT_DIR, 'log', general_args.log_file),
              enable_mail=True,
              mail_setting=mail_setting,
              sub_print=True)




