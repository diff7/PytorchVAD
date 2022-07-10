import os
import torch

torch.set_num_threads(1)


class SileroInferencer:
    def __init__(
        self,
        SAMPLING_RATE,
        silero_window_size=512,
        actual_window_size=160,
        device="cpu",
    ):

        if not os.path.exists(
            "/root/.cache/torch/hub/snakers4_silero-vad_master/files"
        ):
            torch.hub.download_url_to_file(
                "https://models.silero.ai/vad_models/en.wav", "en_example.wav"
            )
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=True,
            onnx=False,
        )

        self.model.to(device)

        (
            get_speech_timestamps,
            save_audio,
            read_audio,
            VADIterator,
            collect_chunks,
        ) = utils

        self.vad_iterator = VADIterator(self.model)
        self.SAMPLING_RATE = SAMPLING_RATE
        self.window_size = silero_window_size
        self.actual_window_size = actual_window_size

    def __call__(self, signal):

        speech_probs = []
        chunks = []
        for i in range(0, len(signal), self.window_size):
            chunk = signal[i : i + self.window_size]
            if len(chunk) < self.window_size:
                break
            chunks.append(chunk.unsqueeze(0))
        chunks = torch.cat(chunks, 0)
        speech_prob = self.model(chunks, self.SAMPLING_RATE)
        with torch.no_grad():
            speech_probs.append(speech_prob.detach())

        self.vad_iterator.reset_states()  # reset model states after each audio

        # aling predictions with 10ms lengths

        speech_probs = torch.cat(speech_probs)
        aligned_probs = speech_probs.repeat_interleave(
            self.window_size // self.actual_window_size, dim=0
        )

        # some error will arise from here but it's ok
        # we will cut half ot the signal because error is accumulating onwards
        return aligned_probs.unsqueeze(0)

