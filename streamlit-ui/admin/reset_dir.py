# Useful to reset directories when uploading so that we can start off from a clean slate

import os
from termcolor import cprint

# Directories to clean: Documents, Chat History, Output, Chunks
def reset_directories():
    directories = ["documents", "chat_history", "output", "chunks", "topic_models_cache"]
    prepend_path_to_dir = "../jarvis-ui/"

    for directory in directories:
        # print(os.path.abspath(prepend_path_to_dir + directory))
        path_to_dir = os.path.abspath(prepend_path_to_dir + directory)
        if os.path.exists(path_to_dir):
            for file in os.listdir(path_to_dir):
                file_path = os.path.join(path_to_dir, file)
                # print(file_path)
                # Check with user before deleting
                print(f"Delete {file_path}? (y/n)")
                user_input = input()
                if user_input.lower() != "y":
                    print(f"Skipping {file_path}")
                    continue
                print(f"Deleting {file_path}")
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        for sub_file in os.listdir(file_path):
                            sub_file_path = os.path.join(file_path, sub_file)
                            os.unlink(sub_file_path)
                        os.rmdir(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
                cprint(f"Deleted {file_path}", "red")
        else:
            print(f"Directory {path_to_dir} does not exist.")

if __name__ == "__main__":
    reset_directories()
    cprint("Directories reset.", "green")