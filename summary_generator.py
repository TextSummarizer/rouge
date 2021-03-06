# coding=utf-8
"""
Il summary generator è un oggetto che si occupa di iterare su una directory piena di testi e di riassumeri,
posizionando tali riassunti in una cartella di destinazione. È necessario fornire in input la lunghezza dei riassunti
"""

import os
import data
import summarizer


def read_len(target_length_path):
    res_map = {}
    f = open(target_length_path, "r")
    lines = f.readlines()

    for line in lines:
        l = line[:len(line) - 1]
        l = l.split(",")
        res_map[l[0]] = int(l[1])

    return res_map


class SummaryGenerator:
    def __init__(self, body_dir_path, target_length_path, destination_path):
        self.body_dir_path = body_dir_path
        self.target_length_path = target_length_path
        self.destination_path = destination_path

    def run(self, summarizer, tfidf_threshold=0.2, redundancy_threshold=0.95):
        # Read summary's target lengths. Store them in a map (file_name -> target_length)
        len_map = read_len(self.target_length_path)

        # Set up the summarizer
        summarizer.set_tfidf_threshold(tfidf_threshold)
        summarizer.set_redundancy_threshold(redundancy_threshold)

        # Iterate over text directory and use the summarizer.py to generate summaries
        for filename in os.listdir(self.body_dir_path):
            summary_length = len_map[filename]
            summary = summarizer.summarize(self.body_dir_path + filename, summary_length)
            data.export_summary(output_dir_path=self.destination_path, filename=filename, text=summary)

    def run_lda(self, summarizer, num_topic, num_words):
        # Read summary's target lengths. Store them in a map (file_name -> target_length)
        len_map = read_len(self.target_length_path)

        # Iterate over text directory and use the summarizer.py to generate summaries
        for filename in os.listdir(self.body_dir_path):
            summary_length = len_map[filename]
            summary = summarizer.summarize(self.body_dir_path + filename, summary_length, "lda", num_topic, num_words)
            data.export_summary(output_dir_path=self.destination_path, filename=filename, text=summary)

"""s = summarizer.Summarizer(model_path="C:/enwiki_20161220_skip_300.bin")
body_dir = 'body/'
target_length = 'len.txt'
destination = 'systems/'
sg = SummaryGenerator(body_dir, target_length, destination)
sg.run(s)"""

