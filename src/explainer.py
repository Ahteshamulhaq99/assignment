import numpy as np
import torch
import logging
from typing import List, Dict, Any, Tuple
from src.config import FASHION_ATTRIBUTES, SIMILARITY_THRESHOLD, LLM_MODEL_NAME
from src.embedder import CLIPEmbedder

logger = logging.getLogger(__name__)

class ExplanationEngine:
    def __init__(self, embedder: CLIPEmbedder):
        self.embedder = embedder
        self.llm_pipeline = None
        
        # Precompute attribute text embeddings
        self.attribute_vocab = FASHION_ATTRIBUTES
        self.attribute_embeddings = {}
        self._precompute_attribute_embeddings()
        
        # Always initialize LLM for generating explanations
        self._initialize_llm()

    def _precompute_attribute_embeddings(self):
        """
        Precompute text embeddings for all fashion attributes with context templates
        to maximize alignment in the CLIP space.
        """
        logger.info("Pre-computing text embeddings for fashion attributes...")
        
        # Define context templates for each group
        templates = {
            "color": "a photo of a {attribute} colored clothing item",
            "pattern": "a photo showing a {attribute} pattern",
            "category": "a photo of a {attribute}",
            "neckline": "a clothing item with a {attribute} neckline",
            "sleeve": "a clothing item with {attribute}s",
            "material": "a photo of a {attribute} fabric garment"
        }
        
        for group, items in self.attribute_vocab.items():
            templated_texts = [templates[group].format(attribute=item) for item in items]
            
            # Embed all templates in this group
            embeddings = self.embedder.embed_texts(templated_texts)
            
            # Store embeddings and original labels
            self.attribute_embeddings[group] = {
                "labels": items,
                "embeddings": embeddings  # shape: (num_items, 512)
            }
            
        logger.info("Fashion attribute embeddings pre-computed successfully.")

    def _initialize_llm(self):
        """
        Loads the local LLM for generating visual search explanations.
        If loading fails, a basic fallback explanation will be used.
        """
        try:
            from transformers import pipeline
            logger.info(f"Loading local explanation LLM: {LLM_MODEL_NAME}...")
            # Load in float16 for memory efficiency on the RTX GPU
            self.llm_pipeline = pipeline(
                "text-generation",
                model=LLM_MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info("Explanation LLM loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load explanation LLM: {e}. Explanations will use a basic fallback.")

    def _get_top_attributes(self, img_embedding: np.ndarray, threshold: float = 0.19) -> Dict[str, List[Tuple[str, float]]]:
        """
        Projects an image embedding onto the attribute text space to find the
        detected visual concepts. Returns matching attributes per group.
        """
        detected = {}
        # Ensure image embedding is 1D
        emb = img_embedding.flatten()
        
        # Enforce limits per category to prevent wordy/conflicting results
        group_limits = {
            "color": 2,
            "pattern": 1,
            "category": 1,
            "neckline": 1,
            "sleeve": 1,
            "material": 1
        }
        
        for group, data in self.attribute_embeddings.items():
            labels = data["labels"]
            attr_embs = data["embeddings"]  # shape: (num_items, 512)
            
            # Dot product (since both are L2 normalized, this is cosine similarity)
            scores = attr_embs @ emb
            
            # Filter and sort attributes
            matches = []
            for score, label in zip(scores, labels):
                if score > threshold:
                    matches.append((label, float(score)))
            
            # Sort by score descending
            matches.sort(key=lambda x: x[1], reverse=True)
            
            # Apply limit
            limit = group_limits.get(group, 1)
            detected[group] = matches[:limit]
            
        return detected

    def generate_explanation(self, 
                             query_emb: np.ndarray, 
                             product_emb: np.ndarray, 
                             product_metadata: Dict[str, Any], 
                             score: float) -> str:
        """
        Generates a plain-text explanation of why the product was retrieved
        using the local LLM (Qwen2.5-1.5B-Instruct).
        """
        # 1. Get model visual observations for query and product
        query_attrs = self._get_top_attributes(query_emb, threshold=0.19)
        product_attrs = self._get_top_attributes(product_emb, threshold=0.19)
        
        # 2. Identify shared observations
        shared_visuals = {}
        for group in self.attribute_vocab.keys():
            q_labels = {item[0] for item in query_attrs[group]}
            p_labels = {item[0] for item in product_attrs[group]}
            overlap = q_labels.intersection(p_labels)
            if overlap:
                shared_visuals[group] = list(overlap)
        
        # 3. Format attributes for LLM context
        q_desc = []
        for g, vals in query_attrs.items():
            if vals:
                q_desc.append(f"{g}: {vals[0][0]}")
        p_desc = []
        for g, vals in product_attrs.items():
            if vals:
                p_desc.append(f"{g}: {vals[0][0]}")
        
        # 4. Generate LLM explanation
        if self.llm_pipeline is not None:
            try:
                prompt = (
                    f"<|im_start|>system\nYou are a helpful fashion assistant. Write a short, natural, non-technical explanation "
                    f"of why the recommended product matches the uploaded query image. "
                    f"Use the shared attributes to explain the visual reasoning in 1-2 friendly sentences. Do not mention technical terms "
                    f"like CLIP, embedding, vector, database, or cosine similarity. Keep it concise.<|im_end|>\n"
                    f"<|im_start|>user\n"
                    f"Query Image attributes detected: {', '.join(q_desc)}.\n"
                    f"Recommended Product attributes: {', '.join(p_desc)}.\n"
                    f"Shared features: {shared_visuals}.\n"
                    f"Similarity Score: {score:.2f}.\n"
                    f"Write the explanation:<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                
                res = self.llm_pipeline(
                    prompt, 
                    max_new_tokens=60, 
                    num_return_sequences=1, 
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.llm_pipeline.tokenizer.eos_token_id
                )
                generated_text = res[0]["generated_text"]
                # Extract assistant reply
                assistant_reply = generated_text.split("<|im_start|>assistant\n")[-1].strip()
                # Clean up any trailing tags
                assistant_reply = assistant_reply.split("<|im_end|>")[0].strip()
                
                return f"{assistant_reply} (Match Strength: {score:.1%})"
            except Exception as e:
                logger.error(f"Error in LLM explanation generation: {e}")
        
        # Minimal fallback if LLM is unavailable
        return f"This product is a strong visual match to your query image. (Match Strength: {score:.1%})"
