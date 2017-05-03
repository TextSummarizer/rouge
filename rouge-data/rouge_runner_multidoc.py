"""
Calcola i valori rouge confrontando i riassunti generati da un summarizer automatico 
(che devono essere messi nella cartella rouge-data/systems/) e i relativi gold standard
(che devono essere messi nella cartella rouge-data/gold_standard)
"""

import rouge
# script_path = 'C:/Users/Peppo/Desktop/w2vm/rouge4MultiLing/rouge/ROUGE-1.5.5.pl'
# data_path = 'C:/Users/Peppo/Desktop/w2vm/rouge4MultiLing/rouge/data'
# system_summary_path = 'C:/evaluation/095/summaries_300_bugfix/'
# gold_standard_path = 'C:/evaluation/summaries_300_500_eval/'
results_path = 'results.txt'
script_path = 'ROUGE-1.5.5.pl'
data_path = 'data'
gold_standard_path = 'models/'
system_summary_path = 'systems/'

print 'Grid search on summaryes with ROUGE metrics: STARTED!'
rouge.compute(results_path=results_path,
              script_path=script_path,
              data_path=data_path,
              system_summary_path=system_summary_path,
              gold_standard_path=gold_standard_path)
print "Everything done. Now go to your result path and see results. Bye!"
