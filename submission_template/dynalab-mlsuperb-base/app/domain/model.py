import numpy as np
from typing import List
from app.domain.schemas.model import ModelSingleInput, ModelSingleOutput, ModelBatchInput
import torch


class ModelController:
    def __init__(self, model_id: str = "path_to_my_checkpoint", device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        self.initialized = True
        self.device = device
        self.pretrained_model = # instaniate your model and load its weights here
        self.sample_rate
        
    def single_inference(self, input_data: ModelSingleInput) -> ModelSingleOutput:
        """Run inference on a single audio sample.
        
        Args:
            input_data: ModelSingleInput containing audio, sample rate, and language if available
            
        Returns:
            ModelSingleOutput with transcription and confidence score
        """
        # Ensure audio is float32
        audio = torch.from_numpy(np.array(input_data.audio).astype(np.float32))
        audio = audio.to(self.device)

        if self.sample_rate != input_data.sample_rate:
            # if your model is not using 16kHz sampling rate
            # you should resample the input audio

        if input_data.language is not None and len(input_data.language) > 0:
           
           # for this utterance, you have access to the true language identity.
           # you can use it as a condition for your model or ignore it
           pred_lid = input_data.language
           pred_asr = 
        else:
            
            # for this utterance, you dont have access to the true language identity.
            # you need to predict it and return it along with the predicted ASR text

            pred_lid = 
            pred_asr = 

        return ModelSingleOutput(
            language=pred_lid,
            text=pred_asr,
        )
    