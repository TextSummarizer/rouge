import summarizer
import summary_generator as sg
import rouge

# Fissare le variabili
body = "rouge-data/body/"
len = "rouge-data/len.txt"
destination = "rouge-data/systems/"
result = "lda_result.txt"
rouge_script = "rouge-data/ROUGE-1.5.5.pl"
data_rouge = "rouge-data/data/"
summary_dest = "rouge-data/systems/"
gold = "rouge-data/models/"

# Preparare il summarizer
s = summarizer.Summarizer(model_path="C:/enwiki_20161220_skip_300.bin")

# Preparare il loop
num_topic = [2, 3, 4, 5, 6, 7, 8, 9]
num_words = [2, 3, 4, 5, 6, 7, 8, 9]

print "Start summarizing..."
for topic in num_topic:
    for word in num_words:
        new_dir = 'topic_' + str(topic) + "_word_" + str(word)
        print new_dir
        destination_path = destination + new_dir

        generator = sg.SummaryGenerator(body_dir_path=body, target_length_path=len, destination_path=destination_path)
        generator.run_lda(s, topic, word)

# Calcolare i rouge
rouge.compute_for_lda(num_topic, num_words, result, rouge_script, data_rouge, summary_dest, gold)
print "Everything done. Now go to your result path and see results. Bye!"
