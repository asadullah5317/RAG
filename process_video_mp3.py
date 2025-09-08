import os
import subprocess
files=os.listdir("videos")
for file in files:
    tutorial_number=file.split("#")[1].split(".")[0]  # type: ignore
    tutorial_name= file.split(" - ")[0] # type: ignore
    result= tutorial_number + " " + tutorial_name  # type: ignore
    print(result) # type: ignore
    subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{result}.mp3"]) # type: ignore