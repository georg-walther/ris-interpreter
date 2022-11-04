import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

input_text = sys.argv[1]

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
translator = pipeline("translation", model=model, tokenizer=tokenizer)
output_text = translator(input_text) # max_length (maximum sequence length) is 512 -> split your text accordingly
print('>>> ' + output_text[0]['translation_text'])