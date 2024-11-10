import os

for i in range(0, 150):
    os.system(f"cargo run --release < in/{i:04}.txt > out/{i:04}.txt")