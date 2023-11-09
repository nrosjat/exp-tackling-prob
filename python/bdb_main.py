# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:05:21 2023

@author: Dr. Nils Rosjat
"""

import subprocess

def run_script(script_name):
    """Execute a python script."""
    subprocess.run(['python', script_name])

if __name__ == "__main__":
    scripts_to_run = ['bdb_feature_extraction.py', 'bdb_model_training.py', 'bdb_analytics.py', 'bdb_plots.py']

    for script in scripts_to_run:
        run_script(script)