import torch.nn as nn

class Vasf(nn.Module):
    def __init__(self, feature_extractor_model, descriptor_model, decoder_model, quantizer=None) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor_model
        self.descriptor = descriptor_model
        self.decoder_model = decoder_model
        self.quantizer = quantizer
    
    def reconstruct(self, x, dsc_len):
        _, _, h, w = x.shape
        feature_map = self.feature_extractor(x)
        descriptor_result = self.descriptor(feature_map, dsc_len)
        descriptor_commit_loss = descriptor_result.get('commitment_loss',None)
        descriptor_output_mask = descriptor_result.get('is_hot',None)
        descriptor_tokens = descriptor_result['tokens']
        vq_commit_loss = None
        if self.quantizer is not None:
            descriptor_tokens, indices, vq_commit_loss = self.quantizer(descriptor_tokens)
        decoder_result = self.decoder_model(descriptor_tokens, image_size=(h,w), token_mask=descriptor_output_mask)
        result = {
            'commit_loss': descriptor_commit_loss,
            'vq_loss':vq_commit_loss,
            'output': decoder_result['output'],
            'token_outputs': decoder_result['token_outputs'],
            'masks': decoder_result['masks']
        }
        return result
        