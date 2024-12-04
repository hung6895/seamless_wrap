import grpc
import pyaudio
import wave
import io
import threading
import queue
import seamless_m4t_pb2
import seamless_m4t_pb2_grpc
import re
import re
import argparse
from collections import Counter
import time
# Global variable to signal program termination
stop_signal = False
start_signal = False
debug = False #False
JUNK_THRESHOLD=10
JUNK_ALERT=3

def count_total_repeats(text):
    """
    Calculates the total repeat times of words that appear more than 5 times in the given text.

    Parameters:
    text (str): The input string to analyze.

    Returns:
    tuple: A tuple containing the total repeat count and a dictionary of words with their counts.
    """
    # Normalize the text to lowercase and extract words
    words = text.split(" ")

    # Count the occurrences of each word
    word_counts = Counter(words)

    # Filter words that occur more than 5 times
    repeated_words = {word: count for word, count in word_counts.items() if count > JUNK_ALERT}

    # Calculate the total repeat times of those words
    total_repeats = sum(repeated_words.values())
    if (debug):
        print(" junk ",text, total_repeats)
    return total_repeats



def record_audio_to_queue(audio_queue, chunk_duration_s, sample_rate=16000, channels=1):
    """
    Captures audio from the sound card and pushes it to a queue in WAV format.
    """
    global start_signal, stop_signal
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    window=4000*4 #8192 #4096

    # Open the audio stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=window)  # Increased buffer size

    if debug:
        print("Recording audio... Press Ctrl+C to stop.")
    while not start_signal:
        pass  # Wait for the start signal
    try:
        while not stop_signal:
            frames = []

            # Record for the specified duration
            for _ in range(0, int(sample_rate /window  * chunk_duration_s)):
                try:
                    data = stream.read(window, exception_on_overflow=False)  # Handle overflow
                    frames.append(data)
                except OSError as e:
                    print(f"Audio buffer overflow: {e}")
                    break

            # Write audio data to WAV format in memory
            with io.BytesIO() as wav_buffer:
                with wave.open(wav_buffer, 'wb') as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(sample_rate)
                    wf.writeframes(b"".join(frames))
                audio_queue.put(wav_buffer.getvalue())

    except KeyboardInterrupt:
        print("Stopped recording.")
    finally:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        p.terminate()
        audio_queue.put(None)  # Signal the end of recording

#audio_queue, server_address, tgt_lang, save_option, output_file
def send_audio_chunks_to_server(audio_queue, server_address, tgt_lang,save_option, output_file):
    """
    Sends audio chunks from the queue to the gRPC server and prints the transcriptions.
    """
    global start_signal, stop_signal
    # Connect to the gRPC server
        # Load the server's certificate
    with open("server.crt", "rb") as cert_file:
        server_cert = cert_file.read()
    
    # Create SSL credentials for the client
    credentials = grpc.ssl_channel_credentials(root_certificates=server_cert)
    # Connect to the gRPC server with a secure channel
    channel = grpc.secure_channel(server_address, credentials)

    #channel = grpc.insecure_channel(server_address)
    stub = seamless_m4t_pb2_grpc.SeamlessM4TServiceStub(channel)
    f=None
    if save_option:
        f= open(output_file, 'a', encoding='utf-8')
    chunk_id = 0
    print("Transribing with "+tgt_lang)
    while not start_signal:
        pass  # Wait for the start signal
    try:
        while not stop_signal:
            audio_data = audio_queue.get()
            if audio_data is None:  # End of recording
                break

            try:
                if debug:
                    print(f"Sending chunk {chunk_id} to server...")
                    start_time=time.time()

                # Create and send the request
                request = seamless_m4t_pb2.SpeechToTextRequest(audio=audio_data, tgt_lang=tgt_lang)
                response = stub.SpeechToText(request)

                # Print the response
                transcribe=response.text
                transcribe = re.sub(r"&nbsp;", " ", transcribe)

                list_key=[]
                junk=False
                if (count_total_repeats(transcribe)>JUNK_THRESHOLD):
                    junk= True
                if (not junk):
                    print(f"{transcribe}")
                    if save_option:
                        f.write(transcribe+"\n")
                        f.flush()
                    chunk_id += 1
                if (debug):
                    print(time.time()-start_time)
            except grpc.RpcError as e:
                print(f"gRPC Error: {e.code()} - {e.details()}")
            except Exception as e:
                print(f"Unexpected error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if save_option:
            f.close()
def monitor_input():
    global start_signal, stop_signal
    print("Press 's' and Enter to start. Press 'q' and Enter to stop.")
    while not stop_signal:
        user_input = input()
        if user_input.strip().lower() == 'q':
            stop_signal = True
            print("Stopping program...")
        if user_input == 's' and not start_signal:
            start_signal = True
            print("Program started.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio transcription client.")
    parser.add_argument("--tgt_lang", type=str, default="eng", help="Target language for transcription (e.g., 'eng', 'vie').")
    parser.add_argument("--save", action="store_true", help="Save the transcription to a file.")
    parser.add_argument("--output_file", type=str, default="transcription.txt", help="File to save the transcription.")
    parser.add_argument("--port", type=str, default="40000", help="Port of server.")
    args = parser.parse_args()

    chunk_duration_s = 8
    port = args.port #"50769" #40000
    server_address = f"reserve.aixblock.io:{port}"
    # Create a queue to share audio chunks between threads
    audio_queue = queue.Queue()

    # Start the recording and sending threads
    recorder_thread = threading.Thread(target=record_audio_to_queue, args=(audio_queue, chunk_duration_s))
    sender_thread = threading.Thread(target=send_audio_chunks_to_server, args=(audio_queue, server_address, args.tgt_lang, args.save, args.output_file))
    input_thread = threading.Thread(target=monitor_input)
    #start
    recorder_thread.start()
    sender_thread.start()
    input_thread.start()
    # Wait for both threads to finish
    recorder_thread.join()
    sender_thread.join()
    input_thread.join()
    print("Recording and transcription completed.")

