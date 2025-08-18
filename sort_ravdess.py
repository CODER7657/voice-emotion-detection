import os
import shutil

def sort_ravdess_files():
    """
    Sorts RAVDESS audio files into happy, sad, and angry categories.
    """
    # --- Configuration ---
    # IMPORTANT: Change this path to where you unzipped the RAVDESS data
    source_path = input("Please enter the full path to the 'Audio_Speech_Actors_01-24' folder: ")

    # This should be the 'data' folder in your project directory
    destination_path = "data"

    if not os.path.isdir(source_path):
        print(f"Error: The source path '{source_path}' does not exist or is not a directory.")
        return

    # Emotion mapping from filename to folder name
    emotion_map = {
        "03": "happy",
        "04": "sad",
        "05": "angry",
    }

    print("Starting file sorting process...")
    copied_count = 0

    # Iterate through all actor folders (e.g., Actor_01, Actor_02, ...)
    for actor_folder in os.listdir(source_path):
        actor_path = os.path.join(source_path, actor_folder)

        if os.path.isdir(actor_path):
            # Iterate through all .wav files in the actor's folder
            for filename in os.listdir(actor_path):
                if filename.endswith(".wav"):
                    try:
                        # Filename format: 03-01-03-02-01-01-12.wav
                        parts = filename.split('-')
                        emotion_code = parts[2]

                        # Check if this emotion is one we want to sort
                        if emotion_code in emotion_map:
                            emotion_folder_name = emotion_map[emotion_code]
                            dest_folder = os.path.join(destination_path, emotion_folder_name)

                            # Create the destination folder if it doesn't exist
                            os.makedirs(dest_folder, exist_ok=True)
                            # Construct full source and destination paths
                            src_file_path = os.path.join(actor_path, filename)
                            dest_file_path = os.path.join(dest_folder, filename)

                            # Copy the file
                            shutil.copy2(src_file_path, dest_file_path)
                            copied_count += 1

                    except IndexError:
                        # Handle filenames that don't match the expected format
                        print(f"Skipping malformed filename: {filename}")

    print(f"\nSorting complete! Copied {copied_count} files successfully.")

if __name__ == "__main__":
    sort_ravdess_files()