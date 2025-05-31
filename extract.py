#!/usr/bin/env python3
"""
Image Text Extraction using Google Gemini API
Processes images from a local directory and extracts text using Gemini API
with OpenAPI 3.0 structured output specification.
"""

import os
import json
import base64
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import mimetypes
import google.generativeai as genai
from PIL import Image
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TextExtraction:
    """Data class for structured text extraction results."""

    filename: str
    extracted_text: Dict[str, Any] | None
    confidence_score: Optional[dict] = None
    language_detected: Optional[dict] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None


class GeminiImageProcessor:
    """Handles image processing and text extraction using Google Gemini API."""

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the Gemini Image Processor.

        Args:
            api_key: Google AI API key
            model_name: Gemini model to use
        """
        self.api_key = api_key
        self.model_name = model_name

        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        logger.info(f"Initialized Gemini processor with model: {model_name}")

    def _is_supported_image(self, file_path: Path) -> bool:
        """Check if file is a supported image format."""
        return file_path.suffix.lower() in self.SUPPORTED_FORMATS

    def _prepare_image(self, image_path: Path) -> Optional[Image.Image]:
        """Load and prepare image for processing."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")

                # Resize if too large (Gemini has size limits)
                max_size = 4096
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    logger.info(f"Resized {image_path.name} to fit within {max_size}px")

                return img.copy()
        except Exception as e:
            logger.error(f"Error preparing image {image_path}: {e}")
            return None

    def _build_openapi_schema(self) -> Dict[str, Any]:

        # OpenAPI 3.0 Schema for structured output
        OPENAPI_SCHEMA = {
            "type": "object",
            "properties": {
                "confidence_score": {
                    "type": "number",
                    # "minimum": 0,
                    # "maximum": 1,
                    "description": "Confidence score for text extraction (0-1)",
                },
                "language_detected": {
                    "type": "string",
                    "description": "Primary language detected in the text",
                },
            },
            "required": ["extracted_text"],
        }

        # import levelling_form.json from /schemas and add to "extracted_text" property
        with open("schemas/levelling_form.json", "r") as f:
            levelling_form_schema = json.load(f)

        OPENAPI_SCHEMA["properties"]["extracted_text"] = levelling_form_schema

        return OPENAPI_SCHEMA

    def _create_extraction_prompt(self) -> str:
        """Create the prompt for text extraction."""
        return """
        Please extract the text from this image and provide a structured response.
        The image is a form that contains handwritten reponses.
        Estimate confidence scores for the extraction quality
        Detect the primary language of the text
        Please format your response as JSON following the provided schema.
        """

    def extract_text_from_image(self, image_path: Path) -> TextExtraction:
        """
        Extract text from a single image using Gemini API.

        Args:
            image_path: Path to the image file

        Returns:
            TextExtraction object with results
        """
        import time

        start_time = time.time()

        try:
            # Prepare image
            image = self._prepare_image(image_path)
            if not image:
                return TextExtraction(
                    filename=image_path.name,
                    extracted_text=None,
                    error="Failed to load or prepare image",
                )

            # Create prompt
            prompt = self._create_extraction_prompt()

            # Create OpenAPI schema for structured output
            openapi_schema = self._build_openapi_schema()

            # Generate content with structured output
            response = self.model.generate_content(
                [prompt, image],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=openapi_schema,
                ),
            )

            # Parse response
            try:
                result_data = json.loads(response.text)
            except json.JSONDecodeError:
                # Fallback: try to extract text without structured format
                logger.warning(
                    f"Failed to parse JSON response for {image_path.name}, using raw text"
                )
                result_data = {"extracted_text": response.text}

            processing_time = time.time() - start_time

            return TextExtraction(
                filename=image_path.name,
                extracted_text=result_data.get("extracted_text", None),
                confidence_score=result_data.get("confidence_score"),
                language_detected=result_data.get("language_detected"),
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            return TextExtraction(
                filename=image_path.name,
                extracted_text=None,
                error=str(e),
                processing_time=time.time() - start_time,
            )

    def process_directory(
        self, input_dir: Path, output_dir: Path
    ) -> List[TextExtraction]:
        """
        Process all images in a directory.

        Args:
            input_dir: Directory containing images
            output_dir: Directory to save results

        Returns:
            List of TextExtraction results
        """
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all supported images
        image_files = [
            f
            for f in input_dir.iterdir()
            if f.is_file() and self._is_supported_image(f)
        ]

        if not image_files:
            logger.warning(f"No supported image files found in {input_dir}")
            return []

        logger.info(f"Found {len(image_files)} images to process")

        results = []
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"Processing {i}/{len(image_files)}: {image_path.name}")

            result = self.extract_text_from_image(image_path)
            results.append(result)

            # Save individual result
            result_file = output_dir / f"{image_path.stem}_extraction.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(asdict(result), f, indent=2, ensure_ascii=False)

            logger.info(f"Saved result to {result_file}")

        # Save combined results
        combined_file = output_dir / "all_extractions.json"
        with open(combined_file, "w", encoding="utf-8") as f:
            json.dump(
                [asdict(result) for result in results], f, indent=2, ensure_ascii=False
            )

        logger.info(f"Saved combined results to {combined_file}")

        return results


def main():
    """Main function to run the image text extraction."""

    # Configuration
    API_KEY = os.getenv("GOOGLE_AI_API_KEY")
    INPUT_DIR = Path("images")
    OUTPUT_DIR = Path("output")
    MODEL_NAME = "gemini-2.0-flash"

    if not API_KEY:
        print("Error: Please set the GOOGLE_AI_API_KEY environment variable")
        print("You can get an API key from: https://aistudio.google.com/app/apikey")
        return

    try:
        # Initialize processor
        processor = GeminiImageProcessor(API_KEY, MODEL_NAME)

        # Process images
        results = processor.process_directory(INPUT_DIR, OUTPUT_DIR)

        # Print summary
        successful = len([r for r in results if not r.error])
        failed = len([r for r in results if r.error])

        print(f"\n--- Processing Complete ---")
        print(f"Total images processed: {len(results)}")
        print(f"Successful extractions: {successful}")
        print(f"Failed extractions: {failed}")

        if successful > 0:
            avg_time = (
                sum(r.processing_time for r in results if r.processing_time)
                / successful
            )
            print(f"Average processing time: {avg_time:.2f} seconds")

        print(f"Results saved to: {OUTPUT_DIR.absolute()}")

    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()
