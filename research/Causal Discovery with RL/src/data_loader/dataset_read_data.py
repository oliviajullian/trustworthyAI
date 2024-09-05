#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" read datasets from existing files"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd


class DataGenerator(object):


    def __init__(self, file_path, solution_path=None, normalize_flag=False, transpose_flag=False):
        
        #TODO: Check for categorical data and apply a mapping
        def map_objects_to_numbers(df):
            # Dictionary to store mappings
            mappings = {}
            
            # Iterate over the columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Create a mapping for this column
                    unique_values = df[col].unique()
                    mapping = {value: float(idx) for idx, value in enumerate(unique_values)}
                    
                    # Apply the mapping to the column
                    df[col] = df[col].map(mapping)
                    
                    # Save the mapping
                    mappings[col] = mapping
            
            # Convert the entire DataFrame to a NumPy array
            numpy_matrix = df.to_numpy()
            
            return numpy_matrix, mappings

        print(file_path)
        if "csv" in file_path:
            self.inputdata = pd.read_csv(file_path)
            self.inputdata, self.mappings = map_objects_to_numbers(self.inputdata)
            print("mappings",self.mappings)
        else:
            self.inputdata = np.load(file_path)
            self.mappings = None


        self.datasize, self.d = self.inputdata.shape

        if normalize_flag:
            self.inputdata = StandardScaler().fit_transform(self.inputdata)

        if solution_path is None:
            gtrue = np.zeros(self.d)
        else:
            gtrue = np.load(solution_path)
            if transpose_flag: 
                gtrue = np.transpose(gtrue)

        print("gtrue", gtrue)

        # (i,j)=1 => node i -> node j
        self.true_graph = np.int32(np.abs(gtrue) > 1e-3)
    
    

    def gen_instance_graph(self, max_length, dimension, test_mode=False):
        seq = np.random.randint(self.datasize, size=(dimension))
        input_ = self.inputdata[seq]
        return input_.T

    # Generate random batch for training procedure
    def train_batch(self, batch_size, max_length, dimension):
        input_batch = []

        for _ in range(batch_size):
            input_= self.gen_instance_graph(max_length, dimension)
            input_batch.append(input_)

        return input_batch
