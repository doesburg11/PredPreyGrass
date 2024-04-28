from pydub import AudioSegment
import os

# Directory containing the MP3 files
directory = "/home/doesburg/Dropbox/Media/Readings/Audiobooks/Ayn Rand/Atlas Shrugged Unabridged"
output = "/home/doesburg/Dropbox/Media/Readings/Audiobooks/Ayn Rand/Atlas Shrugged Unabridged/unabridged.mp3"
# Get a list of all the MP3 files in the directory
mp3_files = [f for f in os.listdir(directory) if f.endswith('.mp3')]

# Sort the files (optional)
mp3_files.sort()

print(f"MP3 files: {mp3_files}")



# Initialize an empty audio segment
combined = AudioSegment.empty()

# Loop over the MP3 files and append them to the combined audio segment
for mp3_file in mp3_files:
    path = os.path.join(directory, mp3_file)
    audio = AudioSegment.from_mp3(path)
    combined += audio
    print(f"Appended {mp3_file}")

# Export the combined audio segment as an MP3 file
print("Exporting combined audio...")
combined.export(output, format='mp3')
print("Export complete.")
