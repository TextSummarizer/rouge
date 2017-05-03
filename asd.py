import summarizer

s = summarizer.Summarizer(model_path="C:/enwiki_20161220_skip_300.bin")
summary = s.summarize("C:/pearljam.txt", 1000, "lda", 3, 3)
print summary

