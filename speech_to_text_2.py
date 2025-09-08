import whisper
import json
import os

model = whisper.load_model("large-v2")
audios = os.listdir("audios")
for audio in audios:
    audio_name = audio.split("_")[0]  # type: ignore # type: ignore (fixes the extraction of number)
    print(audio_name)

    result = model.transcribe(os.path.join("audios", audio), language="hi", task="translate", word_timestamps=False)
    print(result["segments"])  # Print transcription segments

    chunks = []
    for segment in result["segments"]:
        chunks.append({
            "id": segment["id"],
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })
    chunks_with_metadata = {"chunks": chunks, "text": result["text"]}
    
    with open(f"json/{audio_name}.json", "w") as f:
        json.dump(chunks_with_metadata, f)
