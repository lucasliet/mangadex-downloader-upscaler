# MIT License

# Copyright (c) 2022-present Rahman Yusuf

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import os
import subprocess
import threading
import zipfile
from pathlib import Path
import importlib.util
from typing import List

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

NCNN_BINARY_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip"


class NCNNUpscaler:
    def __init__(self, scale: int = 2, concurrency: int = 2):
        self.scale = scale
        self.concurrency = concurrency
        self._shutdown_event = threading.Event()
        self._model_name = "realesrgan-x4plus-anime"

        if scale == 2:
            log.warning(
                f"NCNN upscaler on macOS only supports 4x scale. "
                f"Automatically using 4x instead of {scale}x."
            )
            self.scale = 4

        self.bin_dir = self._get_package_bin_dir()
        self.models_dir = self.bin_dir / "models"
        self.binary_path = self.bin_dir / "realesrgan-ncnn-vulkan"

        self._ensure_binary()
        self._ensure_models()

        log.info(
            f"Real-ESRGAN NCNN initialized (scale={self.scale}x, "
            f"device=ncnn-vulkan, model={self._model_name})"
        )


    def _get_package_bin_dir(self) -> Path:
        """
        Get the correct bin directory for the installed package.
        This ensures binaries are stored in the package location, not the current working directory.
        """
        try:
            spec = importlib.util.find_spec('mangadex_downloader')
            if spec is not None and spec.origin is not None:
                package_path = Path(spec.origin).parent
                log.debug(f"Using package directory: {package_path}")
                return package_path / "bin"
        except Exception as e:
            log.debug(f"Could not determine package location: {e}")
        
        fallback_path = Path(__file__).parent / "bin"
        log.debug(f"Using fallback directory: {fallback_path}")
        return fallback_path
    def _ensure_binary(self):
        if self.binary_path.exists():
            return

        log.info("NCNN binary not found. Downloading...")

        self.bin_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        import urllib.request

        zip_path = self.bin_dir / "ncnn-vulkan.zip"

        try:
            urllib.request.urlretrieve(NCNN_BINARY_URL, zip_path)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    if member.endswith("realesrgan-ncnn-vulkan"):
                        zip_ref.extract(member, self.bin_dir)
                        extracted_path = self.bin_dir / member
                        extracted_path.rename(self.binary_path)
                    elif member.startswith("models/") and member.endswith((".bin", ".param")):
                        member_info = zip_ref.getinfo(member)
                        member_info.filename = Path(member).name
                        zip_ref.extract(member_info, self.models_dir)

            self.binary_path.chmod(0o755)
            log.info(f"NCNN binary and models downloaded to {self.bin_dir}")

        finally:
            if zip_path.exists():
                zip_path.unlink()

    def _ensure_models(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)

        bin_file = self.models_dir / f"{self._model_name}.bin"
        param_file = self.models_dir / f"{self._model_name}.param"

        if not bin_file.exists() or not param_file.exists():
            raise RuntimeError(
                f"Model files not found in {self.models_dir}.\n"
                f"Required: {self._model_name}.bin and {self._model_name}.param\n"
                f"These should have been extracted from the NCNN binary zip.\n"
                f"Try deleting {self.binary_path} and running again to re-download."
            )

    def _is_image(self, path: Path) -> bool:
        return path.suffix.lower() in IMAGE_EXTENSIONS

    def _get_marker_path(self, image_path: str) -> str:
        return f"{image_path}.upscaled"

    def shutdown(self):
        self._shutdown_event.set()

    def _mark_as_upscaled(self, image_path: str, source_hash: str):
        from .format.utils import create_file_hash_sha256

        img_hash = create_file_hash_sha256(image_path)
        marker = self._get_marker_path(image_path)
        with open(marker, 'w') as f:
            f.write(
                f"scale={self.scale}\nmodel={self._model_name}\n"
                f"device=ncnn-vulkan\nhash={img_hash}\nsource_hash={source_hash}\n"
            )

    def _is_already_upscaled(self, image_path: str) -> bool:
        from .format.utils import create_file_hash_sha256

        marker = self._get_marker_path(image_path)
        if not os.path.exists(marker):
            return False

        try:
            with open(marker, 'r') as f:
                content = f.read()

            if f"scale={self.scale}" not in content:
                return False

            stored_hash = None
            for line in content.splitlines():
                if line.startswith("hash="):
                    stored_hash = line.split("=", 1)[1]
                    break

            if not stored_hash:
                return False

            current_hash = create_file_hash_sha256(image_path)
            if current_hash != stored_hash:
                log.debug(
                    f"Hash mismatch for {image_path}, removing marker "
                    f"(stored={stored_hash[:8]}..., current={current_hash[:8]}...)"
                )
                os.remove(marker)
                return False

            return True
        except Exception:
            return False

    def _upscale_single_image(self, input_path: str) -> tuple[str, bool]:
        if self._is_already_upscaled(input_path):
            filename = os.path.basename(input_path)
            log.info(f"Already upscaled: {filename}")
            return (input_path, True)

        from .format.utils import create_file_hash_sha256
        source_hash = create_file_hash_sha256(input_path)

        try:
            ext = Path(input_path).suffix.lower()
            format_map = {'.jpg': 'jpg', '.jpeg': 'jpg', '.png': 'png', '.webp': 'webp'}
            output_format = format_map.get(ext, 'png')

            cmd = [
                str(self.binary_path),
                "-i", input_path,
                "-o", input_path,
                "-n", self._model_name,
                "-s", str(self.scale),
                "-m", str(self.models_dir),
                "-f", output_format
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                log.error(f"NCNN upscale failed for {input_path}: {result.stderr}")
                return (input_path, False)

            self._mark_as_upscaled(input_path, source_hash)
            filename = os.path.basename(input_path)
            log.info(f"Upscaled: {filename}")

            return (input_path, True)

        except subprocess.TimeoutExpired:
            log.error(f"NCNN upscale timeout for {input_path}")
            return (input_path, False)
        except Exception as e:
            log.error(f"Error upscaling {input_path}: {e}")
            return (input_path, False)

    def process_images(self, image_paths: List[str]) -> List[str]:
        if not image_paths:
            return []

        valid_images = [
            img for img in image_paths
            if self._is_image(Path(img))
        ]

        if not valid_images:
            log.debug("No valid images to upscale")
            return image_paths

        log.info(
            f"Upscaling {len(valid_images)} images with Real-ESRGAN NCNN "
            f"(scale={self.scale}x)..."
        )

        failed = []

        try:
            for img in valid_images:
                if self._shutdown_event.is_set():
                    log.info("Upscale cancelled by user")
                    break

                path, success = self._upscale_single_image(img)
                if not success:
                    failed.append(path)

        except KeyboardInterrupt:
            log.info("Upscale interrupted, cancelling pending operations...")
            self._shutdown_event.set()
            raise

        total_failed = len(failed)
        total_success = len(valid_images) - total_failed

        if total_failed > 0:
            log.warning(
                f"Upscale completed: {total_success} successful, "
                f"{total_failed} failed"
            )
        else:
            log.info(
                f"Successfully upscaled {total_success} images with Real-ESRGAN NCNN"
            )

        return image_paths
