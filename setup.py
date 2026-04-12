from setuptools import setup, find_packages

setup(
    name="mediaforge",
    version="1.0.0",
    description="Comprehensive media processing library — video, photo, audio, and AI-powered operations",
    author="MediaForge Team",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "moviepy>=1.0.3",
        "ffmpeg-python>=0.2.0",
        "pydub>=0.25.1",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "loguru>=0.7.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "ai": [
            "openai-whisper>=20231117",
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "transformers>=4.30.0",
            "ultralytics>=8.0.0",
            "rembg>=2.0.50",
        ],
        "ocr": [
            "pytesseract>=0.3.10",
            "easyocr>=1.7.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mediaforge=mediaforge.cli.main:cli",
        ],
    },
)
