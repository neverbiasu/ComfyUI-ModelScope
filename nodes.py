from typing import Optional
from typing import Tuple
import os
import time
import requests
import importlib
import json
import logging

BASE_URL = "https://api-inference.modelscope.cn/v1"

MODELSCOPE_API_KEY = os.getenv("MODELSCOPE_API_KEY") or os.getenv("MODELSCOPE_ACCESS_TOKEN")

DEFAULT_SYSTEM_PROMPT = "You are a helpful and harmless assistant. Answer concisely and helpfully."

# Constants
PLACEHOLDER_MODEL_ID = "Model ID"
PLACEHOLDER_API_KEY = "API Key"
CONTENT_TYPE_JSON = "application/json"

# Logger
logger = logging.getLogger(__name__)


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
            pil_image = self._get_pil_image_module()
            
            # Open image from bytes
            from io import BytesIO
            image = pil_image.open(BytesIO(response.content))
            
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
                        "placeholder": PLACEHOLDER_MODEL_ID,
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
                        "placeholder": PLACEHOLDER_API_KEY,
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
        headers = {"Authorization": f"Bearer {key}", "Content-Type": CONTENT_TYPE_JSON}
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
                        "placeholder": PLACEHOLDER_MODEL_ID,
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
                        "placeholder": PLACEHOLDER_API_KEY,
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
            "Content-Type": CONTENT_TYPE_JSON
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
                        "placeholder": PLACEHOLDER_MODEL_ID,
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
                "steps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Number of denoising steps (higher = better quality, slower).",
                    },
                ),
                "guidance": (
                    "FLOAT",
                    {
                        "default": 3.5,
                        "min": 1.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "How closely to follow the prompt (higher = more adherent).",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 1234,
                        "min": 0,
                        "max": 0xffffffffffffffff,
                        "tooltip": "Random seed for reproducibility.",
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
                "loras": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": '{"lora_id": 0.6} or just lora_id',
                        "tooltip": "LoRA model configuration.",
                    }
                ),
                "api_key": (
                    "STRING",
                    {
                        "placeholder": PLACEHOLDER_API_KEY,
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
        steps: int = 30,
        guidance: float = 3.5,
        seed: Optional[int] = None,
        height: int = 1024,
        width: int = 1024,
        loras: Optional[str] = None,
    ) -> Tuple[object]:
        """Generate image using ModelScope API.
        
        Args:
            model_id: ModelScope model identifier
            prompt: Text description of desired image
            api_key: Optional API key override
            negative_prompt: Elements to avoid in generation
            steps: Number of denoising steps
            guidance: Adherence to prompt
            seed: Random seed
            height: Image height in pixels
            width: Image width in pixels
            loras: LoRA configuration string (ID or JSON)
            
        Returns:
            tuple: Single-element tuple containing image tensor
            
        Raises:
            RuntimeError: If API call or image processing fails
        """
        key = self._resolve_key(api_key)
        # Note: AIGC API uses base_url without /v1/ suffix for the base,
        # but the endpoint is /v1/images/generations.
        # BASE_URL is https://api-inference.modelscope.cn/v1
        # so BASE_URL + "/images/generations" -> .../v1/images/generations
        # which matches the doc: https://api-inference.modelscope.cn/v1/images/generations
        url = f"{BASE_URL}/images/generations"

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": CONTENT_TYPE_JSON,
            "X-ModelScope-Async-Mode": "true"
        }
        
        payload = {
            "model": model_id,
            "prompt": prompt,
            "size": f"{width}x{height}", 
        }

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if steps is not None:
            payload["steps"] = steps
        if guidance is not None:
            payload["guidance"] = guidance
        if seed is not None:
            payload["seed"] = seed

        if loras:
            loras = loras.strip()
            if loras:
                # Try to parse as JSON, otherwise use as string
                try:
                    parsed_loras = json.loads(loras)
                    # Only accept JSON objects (dict); otherwise, use the original string
                    if isinstance(parsed_loras, dict):
                        payload["loras"] = parsed_loras
                    else:
                        payload["loras"] = loras
                except json.JSONDecodeError:
                    # Fallback to string if parsing fails
                    payload["loras"] = loras

        try:
            # Use ensure_ascii=False for Chinese characters support
            data_bytes = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            headers_with_charset = headers.copy()
            headers_with_charset["Content-Type"] = "application/json; charset=utf-8"
            resp = requests.post(url, headers=headers_with_charset, data=data_bytes, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Network error calling ModelScope Image API: {e}") from e

        try:
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Failed to parse API response as JSON: {e}") from e

        task_id = data.get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id returned from async request: {data}")

        # Poll for status
        # BASE_URL is .../v1
        # Task endpoint: .../v1/tasks/{task_id}
        task_url = f"{BASE_URL}/tasks/{task_id}"
        poll_headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": CONTENT_TYPE_JSON,
            "X-ModelScope-Task-Type": "image_generation"
        }

        # Poll for up to 10 minutes (600s)
        max_wait_time = 600
        start_poll_time = time.time()

        while time.time() - start_poll_time < max_wait_time:
            try:
                result = requests.get(task_url, headers=poll_headers, timeout=30)
                result.raise_for_status()
                task_data = result.json()

                status = task_data.get("task_status")
                if status == "SUCCEED":
                    output_images = task_data.get("output_images")
                    if output_images and len(output_images) > 0:
                        image_url = output_images[0]
                        download_headers = {"Authorization": f"Bearer {key}"} if key else None
                        return (self._download_image_from_url(image_url, download_headers),)
                    else:
                        raise RuntimeError(
                            f"Task succeeded but no output images found (status={status}, "
                            f"output_count={len(output_images) if output_images is not None else 0})"
                        )

                elif status == "FAILED":
                    error_message = task_data.get("error_message") or task_data.get("message") or "Unknown error"
                    error_code = task_data.get("error_code") or task_data.get("code")
                    details = f", code={error_code}" if error_code is not None else ""
                    raise RuntimeError(f"Image generation failed (status={status}{details}): {error_message}")

                elif status in ["PENDING", "RUNNING"]:
                    time.sleep(5)
                    continue

                else:
                    # Unknown status: fail fast with a clear error
                    raise RuntimeError(f"Unknown task status '{status}' received from API: {task_data}")

            except requests.RequestException as e:
                # Distinguish permanent HTTP errors (4xx) from transient issues.
                status_code = getattr(getattr(e, "response", None), "status_code", None)
                if status_code is not None and 400 <= status_code < 500:
                    # Client errors are typically permanent for this task; fail fast.
                    raise RuntimeError(
                        f"Image generation polling failed with client error {status_code}: "
                        f"{getattr(e.response, 'text', '')}"
                    ) from e

                # Transient network or server error during polling: log and retry.
                logger.warning(f"Polling failed: {e}. Retrying...")
                time.sleep(5)

        raise RuntimeError("Image generation timed out after polling.")


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
