from typing import Optional
from typing import Tuple
import os
import time
import requests
import importlib

BASE_URL = "https://api-inference.modelscope.cn/v1"

MODELSCOPE_API_KEY = os.getenv("MODELSCOPE_API_KEY") or os.getenv("MODELSCOPE_ACCESS_TOKEN")

DEFAULT_SYSTEM_PROMPT = "You are a helpful and harmless assistant. Answer concisely and helpfully."


class ModelScopeBase:
    """Base class for ModelScope nodes with common functionality."""
    
    def _resolve_key(self, api_key: Optional[str]) -> str:
        """Resolve API key from input or environment variables."""
        key = api_key or MODELSCOPE_API_KEY
        if not key:
            raise ValueError(
                "API key missing. Provide 'api_key' or set MODELSCOPE_API_KEY / MODELSCOPE_ACCESS_TOKEN."
            )
        return key

    def _sanitize_text(self, text: str) -> str:
        """Sanitize text for ComfyUI showText node compatibility.
        
        Args:
            text: Raw text string
            
        Returns:
            str: Sanitized text safe for display
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Remove or replace problematic characters
        import re
        
        # Remove null bytes and other control characters except common ones
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure text is not empty
        if not text:
            text = "[Empty Response]"
        
        # Limit length if extremely long (optional safeguard)
        max_length = 10000
        if len(text) > max_length:
            text = text[:max_length] + "...[truncated]"
        
        # Ensure proper encoding
        try:
            text.encode('utf-8').decode('utf-8')
        except UnicodeError:
            # Fallback: replace problematic characters
            text = text.encode('utf-8', errors='replace').decode('utf-8')
        
        return text

    def _get_pil_image_module(self):
        """Get PIL Image module with lazy import."""
        try:
            return importlib.import_module("PIL.Image")
        except Exception:
            raise RuntimeError("Pillow is required. Install with: pip install pillow")

    def _download_image_from_url(self, url: str, headers: dict) -> object:
        """Download image from URL and convert to ComfyUI IMAGE format.
        
        Args:
            url: Image URL to download
            headers: HTTP headers for authentication
            
        Returns:
            torch.Tensor: Image tensor in ComfyUI format [B, H, W, C]
            
        Raises:
            RuntimeError: If download or conversion fails
        """
        try:
            import torch
            import numpy as np
            
            # Download image
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Get PIL Image module
            PIL_Image = self._get_pil_image_module()
            
            # Open image from bytes
            from io import BytesIO
            image = PIL_Image.open(BytesIO(response.content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert PIL to numpy array
            image_np = np.array(image).astype(np.float32) / 255.0
            
            # Convert to torch tensor with ComfyUI format [B, H, W, C]
            image_tensor = torch.from_numpy(image_np)[None,]
            
            return image_tensor
            
        except Exception as e:
            raise RuntimeError(f"Failed to download image: {e}") from e


class ModelScopeChatBase(ModelScopeBase):
    """Base class for chat-based ModelScope nodes."""
    
    def _extract_text(self, data: dict) -> str:
        """Extract text content from chat completion response.
        Tries to read choices[0].message.content (string or list) and falls back to str(data).
        """
        try:
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, str):
                return self._sanitize_text(content)
            if isinstance(content, list) and content:
                part = content[0]
                if isinstance(part, str):
                    return self._sanitize_text(part)
                if isinstance(part, dict):
                    txt = part.get("text") or part.get("content")
                    if txt:
                        return self._sanitize_text(txt)
        except Exception:
            pass
        # Fallback to string representation
        try:
            return self._sanitize_text(str(data))
        except Exception:
            return "[Error: Unable to extract response]"


class ModelScopeLLM(ModelScopeChatBase):
    """ModelScope Text Chat (non-streaming) node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": (
                    "STRING",
                    {
                        "default": "Qwen/Qwen3-235B-A22B",
                        "placeholder": "Model ID",
                        "tooltip": "The ModelScope model ID to use.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "User prompt",
                        "tooltip": "The user prompt text to send to the model.",
                    },
                ),
            },
            "optional": {
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "System Prompt",
                        "tooltip": "Optional system prompt to prime the assistant.",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "placeholder": "API Key",
                        "tooltip": "ModelScope API key; if empty, read from MODELSCOPE_API_KEY env var.",
                    },
                ),
                "request_timeout": (
                    "INT",
                    {
                        "default": 30,
                        "min": 10,
                        "max": 300,
                        "step": 10,
                        "tooltip": "Request timeout in seconds. Increase for larger models or slow networks.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "chat_completion"
    CATEGORY = "ModelScope"
    DESCRIPTION = "Generate a chat response using ModelScope LLM."
    OUTPUT_TOOLTIPS = ("ModelScope text response.",)

    def chat_completion(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout: int = 30,
    ) -> Tuple[str]:
        """Execute text generation using ModelScope LLM.
        
        Args:
            model_id: ModelScope LLM model identifier
            prompt: User text prompt for the model
            system_prompt: Optional system prompt for assistant behavior
            api_key: Optional API key override
            request_timeout: Request timeout in seconds
            
        Returns:
            tuple: Single-element tuple with LLM response text
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If API call or processing fails
        """
        # Validate inputs
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        key = self._resolve_key(api_key)
        system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        url = f"{BASE_URL}/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                },
            ],
            "stream": False,
            "enable_thinking": False,
        }

        start_time = time.time()
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=request_timeout,
                verify=False,  # Temporary SSL workaround
            )
            resp.raise_for_status()
        except requests.Timeout as e:
            elapsed = time.time() - start_time
            raise RuntimeError(
                f"ModelScope LLM request timed out after {elapsed:.1f}s. "
                f"Try increasing request_timeout (current: {request_timeout}s) or using a smaller model."
            ) from e
        except requests.RequestException as e:
            raise RuntimeError(f"Network error calling ModelScope LLM: {e}") from e

        try:
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Failed to parse LLM API response: {e}") from e

        try:
            text = self._extract_text(data)
            if not text.strip():
                raise RuntimeError("LLM returned empty response")
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from LLM response: {e}") from e

        elapsed = time.time() - start_time
        print(f"LLM inference completed in {elapsed:.1f}s")
        
        return (text,)


class ModelScopeVLM(ModelScopeChatBase):
    """ModelScope Visual Language Model node for image-text conversations.
    
    Supports analyzing images with text prompts using ModelScope VLM models.
    Handles various image input formats and provides robust error handling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define input types and validation rules for VLM processing.
        
        Returns:
            dict: Input type definitions with validation parameters
        """
        return {
            "required": {
                "model_id": (
                    "STRING",
                    {
                        "default": "Qwen/QVQ-72B-Preview",
                        "placeholder": "Model ID",
                        "tooltip": "ModelScope VLM model ID.",
                    },
                ),
                "image_url": (
                    "STRING",
                    {
                        "placeholder": "https://example.com/image.jpg",
                        "tooltip": "URL of the image to analyze. Must be publicly accessible.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "What do you see in this image?",
                        "tooltip": "Question or instruction about the image.",
                    },
                ),
            },
            "optional": {
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "You are a helpful vision assistant...",
                        "tooltip": "Optional system prompt to guide the assistant's behavior.",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "placeholder": "API Key",
                        "tooltip": "ModelScope API key. If empty, reads MODELSCOPE_API_KEY or MODELSCOPE_ACCESS_TOKEN from environment.",
                    },
                ),
                "request_timeout": (
                    "INT",
                    {
                        "default": 120,
                        "min": 30,
                        "max": 300,
                        "step": 10,
                        "tooltip": "Request timeout in seconds. VLM inference can be slow, recommend 120+ seconds.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "visual_chat"
    CATEGORY = "ModelScope"
    DESCRIPTION = "Analyze images with text using ModelScope Vision-Language Models."
    OUTPUT_TOOLTIPS = ("VLM response describing or answering about the image.",)

    def visual_chat(
        self,
        model_id: str,
        image_url: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout: int = 120,
    ) -> Tuple[str]:
        """Execute vision-language model inference on image and text.
        
        Args:
            model_id: ModelScope VLM model identifier
            image_url: URL of image to analyze
            prompt: Text prompt/question about the image
            system_prompt: Optional system prompt for assistant behavior
            api_key: Optional API key override
            request_timeout: Request timeout in seconds
            
        Returns:
            tuple: Single-element tuple with VLM response text
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If API call or processing fails
        """
        # Validate inputs
        if not image_url.strip():
            raise ValueError("Image URL cannot be empty")
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
            
        # Validate image URL format
        if not (image_url.startswith('http://') or image_url.startswith('https://')):
            raise ValueError("Image URL must be a valid HTTP/HTTPS URL")

        key = self._resolve_key(api_key)
        system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        
        url = f"{BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        
        # Construct VLM message payload with image and text
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                },
            ],
            "stream": False,
            "enable_thinking": False,
        }

        start_time = time.time()
        try:
            # Use longer timeout for VLM inference
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=request_timeout,
                verify=False,  # Temporary SSL workaround
            )
            resp.raise_for_status()
        except requests.Timeout as e:
            elapsed = time.time() - start_time
            raise RuntimeError(
                f"ModelScope VLM request timed out after {elapsed:.1f}s. "
                f"Try increasing request_timeout (current: {request_timeout}s) or using a smaller model."
            ) from e
        except requests.RequestException as e:
            raise RuntimeError(f"Network error calling ModelScope VLM: {e}") from e

        try:
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Failed to parse VLM API response: {e}") from e

        try:
            text = self._extract_text(data)
            if not text.strip():
                raise RuntimeError("VLM returned empty response")
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from VLM response: {e}") from e

        elapsed = time.time() - start_time
        print(f"VLM inference completed in {elapsed:.1f}s")
        
        return (text,)


class ModelScopeImageGenerator(ModelScopeBase):
    """ModelScope Image Generation node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": (
                    "STRING",
                    {
                        "default": "Qwen/Qwen-Image-Edit",
                        "placeholder": "Model ID",
                        "tooltip": "ModelScope model ID for image generation.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "A mysterious girl walking down the corridor.",
                        "tooltip": "Text prompt for image generation.",
                    },
                ),
            },
            "optional": {
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "lowres, bad anatomy...",
                        "tooltip": "Negative prompt to avoid unwanted elements.",
                    },
                ),
                "num_inference_steps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Number of denoising steps (higher = better quality, slower).",
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 3.5,
                        "min": 1.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "How closely to follow the prompt (higher = more adherent).",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 2048,
                        "step": 8,
                        "tooltip": "Generated image height in pixels.",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 2048,
                        "step": 8,
                        "tooltip": "Generated image width in pixels.",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "placeholder": "API Key",
                        "tooltip": "ModelScope API key. If empty, reads from MODELSCOPE_API_KEY env var.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_generation"
    CATEGORY = "ModelScope"
    DESCRIPTION = "Generate images using ModelScope text-to-image models."
    OUTPUT_TOOLTIPS = ("Generated image as ComfyUI IMAGE tensor.",)

    def image_generation(
        self,
        model_id: str,
        prompt: str,
        api_key: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 3.5,
        height: int = 1024,
        width: int = 1024,
    ) -> Tuple[object]:
        """Generate image using ModelScope API.
        
        Args:
            model_id: ModelScope model identifier
            prompt: Text description of desired image
            api_key: Optional API key override
            negative_prompt: Elements to avoid in generation
            num_inference_steps: Number of denoising steps
            guidance_scale: Adherence to prompt
            height: Image height in pixels
            width: Image width in pixels
            
        Returns:
            tuple: Single-element tuple containing image tensor
            
        Raises:
            RuntimeError: If API call or image processing fails
        """
        key = self._resolve_key(api_key)
        url = f"{BASE_URL}/images/generations"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "size": f"{width}x{height}", 
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60, verify=False)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Network error calling ModelScope Image API: {e}") from e

        try:
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Failed to parse API response: {e}") from e

        # Handle direct image URL response
        if data.get("images"):
            first_image = data["images"][0]
            if isinstance(first_image, dict) and first_image.get("url"):
                return (self._download_image_from_url(first_image["url"], headers),)

        raise RuntimeError(f"Unable to locate generated image in response: {data}")


NODE_CLASS_MAPPINGS = {
    "ModelScopeLLM": ModelScopeLLM,
    "ModelScopeVLM": ModelScopeVLM,
    "ModelScopeImageGenerator": ModelScopeImageGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelScopeLLM": "ModelScope LLM",
    "ModelScopeVLM": "ModelScope VLM",
    "ModelScopeImageGenerator": "ModelScope Image Generator",
}