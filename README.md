Audio Transcription Client
This Python application records audio from your microphone, sends it to a gRPC server for transcription, and outputs the transcriptions. The user can control the start and stop of the program with specific key inputs (s to start, q to stop). The transcriptions can either be printed to the console or saved to a file.

Features
Record Audio: Captures audio from the system microphone in chunks.
Send to Server: Transmits audio to a gRPC server for transcription.
Transcription: Outputs transcriptions of the audio in the desired target language (eng or vie).
Stop Words: Filters out unwanted phrases or noise words.
Start/Stop Control: Users can start and stop the program manually using s and q.
Save Option: Optionally save transcriptions to a file.
Installation
Prerequisites
Python 3.7+

Install the required Python libraries:

bash
Copy code
pip install grpcio pyaudio
server.crt file: Ensure you have the server certificate in the same directory as the script.

Clone Repository
Download or clone the repository:

bash
Copy code
git clone <repository_url>
cd <repository_directory>
Usage
Run the script using:

bash
Copy code
python h51_client.py [OPTIONS]
Options:
--tgt_lang: Target language for transcription (eng for English, vie for Vietnamese). Default: eng.
--save: Save the transcription to a file. Default: Disabled.
--output_file: File to save the transcription. Default: transcription.txt.
Example Commands
Print transcription to console in English:

bash
Copy code
python h51_client.py --tgt_lang eng
Save transcription to a file:

bash
Copy code
python h51_client.py --tgt_lang vie --save --output_file my_transcription.txt
How It Works
Start the Program:

When the program starts, it waits for the user to press s and hit Enter to begin recording and transcription.
Stop the Program:

Press q and hit Enter to stop the program.
Output:

Transcriptions are either printed to the console or saved to the specified file.
Key Features and Design
Start/Stop Signals
Press s to start the program and begin recording audio.
Press q to stop the program gracefully.
Junk Filtering
Certain phrases or noise words are filtered from the transcription results to improve output quality. 

Known Issues
Microphone Access: Ensure the program has access to the system microphone.
gRPC Server Connection: The server.crt file must be valid, and the server should be reachable at the specified address.
Contributing
Contributions are welcome! Fork the repository and submit a pull request for improvements.

License
This project is licensed under the MIT License.

