import easyocr
from utils.text_extractor import TextExtractor

reader = easyocr.Reader(['en'])

text_extractor = TextExtractor(reader)
