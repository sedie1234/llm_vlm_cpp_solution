from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./vlm/tokenizer")
tokenizer.add_tokens(["<image>"], special_tokens=True)

tokenizer.save_pretrained("./vlm/merged_tokenizer", legacy_format=False)
