from espnet2.bin.asr_inference import Speech2Text
import numpy as np
from typing import List
from app.domain.schemas.model import ModelSingleInput, ModelSingleOutput, ModelBatchInput
import torch


class ModelController:
    def __init__(self, model_id: str = "espnet/mms_1b_mlsuperb", device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        self.initialized = True
        self.device = device
        self.pretrained_model = Speech2Text.from_pretrained(
            model_id,
            device=device
        )
        
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

        if input_data.language is not None and len(input_data.language) > 0:
            # MMS baseline does not use language input
            pass
        
        result = self.pretrained_model(audio)
        text_str, tokens, token_ids, hypothesis = result[0]

        text_str = text_str.split(" ", 1)
        lid= text_str[0]
        text = text_str[1]
    
        return ModelSingleOutput(
            language=lid,
            text=text,
        )
    
    def batch_inference(self, input_data: ModelBatchInput) -> List[ModelSingleOutput]:
        """Run inference on multiple audio samples.
        
        Args:
            input_data: ModelBatchInput containing list of audio samples
            
        Returns:
            List of ModelSingleOutput objects
        """
        predictions = []
        for sample in input_data.dataset_samples:
            single_input = ModelSingleInput(
                audio=sample['audio'],
                sample_rate=sample['sample_rate'],
                language=sample['language']
            )
            predictions.append(self.single_inference(single_input))
        return predictions
    