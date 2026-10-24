import re

# 1. Load the raw text
with open("Alice.txt", encoding="utf-8-sig") as f:
    text = f.read()
print("Original characters:", len(text))

# 2. Cut off the Gutenberg header and footer
start = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text)
end = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK", text)
if start and end:
    text = text[start.end():end.start()]
else:
    print("Warning: could not find the Gutenberg markers, header/footer not removed")

# Skip the title page and table of contents: start at the 2nd "CHAPTER I."
starts = [m.start() for m in re.finditer(r"CHAPTER I\.", text)]
if len(starts) >= 2:
    text = text[starts[1]:]

# 3. Tidy up line endings and extra blank lines
text = text.replace("\r\n", "\n").strip()
text = re.sub(r"\n{3,}", "\n\n", text)

# 4. Report and save
print("Cleaned characters:", len(text))
print("Unique characters:", len(set(text)))
print("\nFirst 300 characters:\n")
print(text[:300])

with open("clean.txt", "w", encoding="utf-8") as f:
    f.write(text)
print("\nSaved clean.txt")