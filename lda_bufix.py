import summarizer


def gen_centroid_lda(lookup_table, sentences, num_topics, num_words):
    from gensim import models, corpora

    # Find topic and probability distribution for each topic (with LDA)
    dictionary = corpora.Dictionary(sentences)  # Usage: remember (id -> term) mapping
    corpus = [dictionary.doc2bow(text) for text in sentences]  # build matrix (corpus is the matrix)
    lda_model = models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
    show_topics = lda_model.show_topics()

    # For each topic, I want to get only "num_words" term (the most relevant, based on probability distribution)
    # Selected token will be part of my "centroid_set"
    # Warning: I have to filter out relevant words that are not in w2v model
    centroid_set = []
    for topic in show_topics:
        token_probability_records = topic[1].split(" + ")
        i = 0
        topic_word_counter = 0
        stop = False

        while i < len(token_probability_records) and not stop:
            # String manipulation (probability distribution is in string format)
            next_record = token_probability_records[i]
            split_record = next_record.split("*")
            token = split_record[1].replace("\"", "")  # (split_record[0] is probability of the token)

            if token not in centroid_set:
                if not lookup_table.unseen(token):
                    centroid_set.append(token)
                    topic_word_counter += 1
                    if topic_word_counter == num_words:
                        stop = True
            i += 1

    top_words_vectorized = map(lambda word: lookup_table.vec(word), centroid_set)
    return sum(top_words_vectorized) / len(top_words_vectorized)


s = summarizer.Summarizer(model_path="C:/enwiki_20161220_skip_300.bin")
filename = "C:/pearljam.txt"
model = s.lookup_table
sentences_for_text, sentences_for_centroid = s._preprocessing_for_lda(filename, True)
centroid = gen_centroid_lda(model, sentences_for_centroid, 3, 3)
print centroid
