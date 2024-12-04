import os
import torch
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model
import seamless_m4t_pb2
import seamless_m4t_pb2_grpc
from concurrent import futures
import grpc
import io
import re

# Set Hugging Face cache path
os.environ["HF_HOME"] = "/path/to/custom/huggingface_cache"  # Change this to your desired path

# Load the transcription model and processor
MODEL_NAME = "facebook/seamless-m4t-v2-large"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model {MODEL_NAME} on {device}...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = SeamlessM4Tv2Model.from_pretrained(MODEL_NAME).to(device)
tgt_lang="vie"
text_tgt_lang_id = model.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
print(text_tgt_lang_id)

tgt_lang="eng"
text_tgt_lang_id = model.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
print(text_tgt_lang_id)

#text_decoder_input_ids = torch.tensor([[text_tgt_lang_id]] * batch_size).to(self.device)



import time
def is_vietnamese(text):
    # Define a regex pattern for Vietnamese-specific characters
    vietnamese_pattern = r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]"
    return bool(re.search(vietnamese_pattern, text))
class SeamlessM4TServicer(seamless_m4t_pb2_grpc.SeamlessM4TServiceServicer):
    def save_audio_to_tmp(self, audio_data, sample_rate=16000, file_format="wav"):
        """
        Save audio data to a temporary file.

        Parameters:
            audio_data (bytes): The audio data from the gRPC request.
            sample_rate (int): The sample rate of the audio.
            file_format (str): The format to save the file (e.g., 'wav').

        Returns:
            str: The path to the temporary audio file.
        """
        try:
            # Create a temporary file
            #tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_format}")
            tmp_file_path = "fail.wav"
            #tmp_file.close()  # Close the file descriptor as we'll write to it below

            # Save the audio data to the temporary file
            with open(tmp_file_path, "wb") as f:
                f.write(audio_data)

            print(f"Audio saved to temporary file: {tmp_file_path}")
            exit(0)
            return tmp_file_path
        except Exception as e:
            print(f"Error saving audio to file: {e}")
            return None
    def SpeechToText(self, request, context):
        """
        Handles a unary SpeechToText request, processes the audio, and returns a transcription.
        """
        try:
            # Load audio from the request directly into memory
            print("Processing received audio in memory...")
            start_time=time.time()
            audio_data = torch.frombuffer(request.audio, dtype=torch.float32)
            
            # Convert the buffer to a waveform tensor
            waveform, sampling_rate = torchaudio.load(io.BytesIO(request.audio), format="wav")
            print(f"Loaded audio: shape={waveform.shape}, sampling_rate={sampling_rate}")

            # Resample to 16 kHz if necessary
            if sampling_rate != 16000:
                print("Resampling audio to 16kHz...")
                resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
                waveform = resampler(waveform)

            # Convert stereo to mono if necessary
            if waveform.shape[0] > 1:
                print("Converting audio to mono...")
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            # Generate transcription
            print(request.tgt_lang)

            # Prepare the input for the model
            inputs = processor(
                audios=waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).to(device)
            tgt_lang="vie"
            text_tgt_lang_id = model.generation_config.text_decoder_lang_to_code_id.get(request.tgt_lang)
            print(text_tgt_lang_id)
            batch_size=1
            text_decoder_input_ids = torch.tensor([[text_tgt_lang_id]] * batch_size).to(device)
            output_tokens = model.generate(**inputs, text_decoder_input_ids=text_decoder_input_ids, generate_speech=False,num_beams=8, speech_do_sample=True,speech_temperature=0.9)
            transcribed_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
            
            print(time.time()-start_time)
            print(f"Transcription result: {transcribed_text}")
            # Return the response
            return seamless_m4t_pb2.SpeechToTextResponse(text=transcribed_text)

        except Exception as e:
            print(f"Error in SpeechToText: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return seamless_m4t_pb2.SpeechToTextResponse(text="Error during transcription.")


def serve():
    """
    Start the gRPC server and listen for client connections.
    """
    """
    Start the gRPC server with secure communication using TLS.
    """
    # Load the server's certificate and private key
    with open("server.crt", "rb") as cert_file, open("server.key", "rb") as key_file:
        server_credentials = grpc.ssl_server_credentials(
            [(key_file.read(), cert_file.read())]
        )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    seamless_m4t_pb2_grpc.add_SeamlessM4TServiceServicer_to_server(SeamlessM4TServicer(), server)
    server.add_secure_port("[::]:9080",server_credentials)
    print("Server is running on port 9080...")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()

