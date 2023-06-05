from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# Step 1: Load pre-trained BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Step 2: Prepare input text
input_text = '''
In these times, it’s hard to write. It’s difficult to focus. Simply put, it’s not a good time.

To overcome writer’s block, I reviewed several prompts including one suggesting to use Wikipedia’s Random Article feature to spark ideas. When I click the “random article” link in Wikipedia’s sidebar, I decide it will be more entertaining to show the wild breadth of topics instead. These ten entries appear in order:

Maisach station
2018–19 CSA 4-Day Franchise Series
Shagun Chowdhary
Gemmotheres
MLS Cup 1998
Instrumentals (Mouse on Mars album)
Blackfoot Mountains
Yalpara Conservation Park
Empire State Development Corporation
Odonatoptera
I can see how a few of those topics might inspire creative writing. Imagine if the constraint was to write a story using 10 random topics!
'''

# Step 3: Tokenize input text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Step 4: Generate summary
summary_ids = model.generate(input_ids, num_beams=4, max_length=100, early_stopping=True)
summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

print("Original Text:")
print(input_text)
print("\nSummary:")
print(summary)