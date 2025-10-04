# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Install core dependencies
pip install -r requirements.txt

# Install optional dependencies (REQUIRED for upscaling, EPUB, CB7 support)
pip install -r requirements-optional.txt

# Install in editable mode with all extras (RECOMMENDED for development)
uv pip install --upgrade ".[optional]"
```

**CRITICAL**: After ANY code changes, you MUST reinstall the package:
```bash
uv pip install --upgrade ".[optional]"
```

Python caches imported modules, so changes won't take effect without reinstallation.

### Running the Application
```bash
# Via installed command
mangadex-dl "https://mangadex.org/..."

# Via Python module
python3 -m mangadex_downloader "https://mangadex.org/..."
```

### Building Executables
```bash
# Build standalone executable using PyInstaller
pyinstaller mangadex-dl_x64.spec  # For 64-bit systems
pyinstaller mangadex-dl_x86.spec  # For 32-bit systems

# Or use compile.py
python compile.py
```

### Code Quality
```bash
# Run ruff linter (target: Python 3.10+, line length: 92)
ruff check .
```

## Architecture Overview

### Pipeline-Based Architecture

The application follows a pipeline pattern: URL → Fetch Metadata → Download Images → Upscale (optional) → Convert to Format → Save

```
CLI Entry Point (cli/download.py)
    ↓
Main Orchestrator (main.py)
    ↓
Fetcher (fetcher.py) ←→ Network Layer (network.py) ←→ MangaDex API
    ↓
Format Classes (format/*.py)
    ↓
├─→ Download Loop (BaseFormat.save_chapter_images)
│   ├─→ ChapterPageDownloader (downloader.py)
│   ├─→ Hash Verification (format/utils.py)
│   └─→ Upscaler (upscale.py) [if enabled]
    ↓
Format Conversion (PDF/EPUB/CBZ/etc.)
    ↓
Final Output File
```

### Key Architectural Patterns

**1. Format System (Inheritance Hierarchy)**
- `BaseFormat`: Abstract base defining the download pipeline
- `BaseConvertedFormat`: Adds format conversion capabilities
- `ConvertedChaptersFormat`, `ConvertedVolumesFormat`, `ConvertedSingleFormat`: Split by grouping type
- Concrete formats (e.g., `ComicBookArchive`, `Epub`, `PDF`) inherit and implement conversion

**2. Lazy Imports for Circular Dependency Prevention**
Import statements are placed inside functions/methods (not at module level) to break circular dependencies between `downloader.py`, `upscale.py`, `network.py`, and `config/config.py`.

Example pattern:
```python
def process_images(self, image_paths):
    from .config import config  # Lazy import
    # ... processing logic using config
```

**3. Async Background Processing**
The `QueueWorker` class handles asynchronous tasks (like sending download reports to MangaDex) in a background thread, preventing blocking of the main download pipeline.

**4. Configuration System**
- `_Config`: Low-level class managing JSON config file with thread-safe locks
- `ConfigProxy`: High-level attribute-style access (`config.save_as`)
- Each option has a validator function
- CLI arguments override config file values

## Critical Implementation Details

### Real-ESRGAN Upscaling Feature

The application supports AI-based image upscaling with platform-specific backends:

**Backend Selection (upscale.py:284):**
- **macOS (Darwin)**: Core ML with Apple Neural Engine (`upscale_coreml.py`)
- **Linux/Windows**: PyTorch with CUDA/CPU (`upscale.py`)

#### Core ML Backend (macOS Only)

Located in `upscale_coreml.py`. Uses Apple's Core ML framework with Neural Engine acceleration.

**Key features:**
1. **Model**: `RealESRGAN_x2plus.mlpackage` (2x native upscaling)
   - Downloaded automatically from `http://upscale.aidoku.app/models/RealESRGAN_x2plus.mlpackage.zip`
   - Stored in `mangadex_downloader/models/`
   - Size: ~50MB compressed

2. **Neural Engine Optimization**:
   - Uses `coremltools.ComputeUnit.ALL` for hardware acceleration
   - Zero CPU/GPU usage during upscaling (Activity Monitor won't show activity)
   - Automatic model compilation and caching by Core ML

3. **Image Processing**:
   - Direct PIL Image input/output (no numpy conversion needed)
   - Preserves original image format (JPEG quality=95, PNG optimized)
   - Handles RGB and RGBA modes automatically

4. **Requirements**:
   - macOS 12+ (Monterey or newer)
   - Apple Silicon (M1/M2/M3/M4) recommended for Neural Engine
   - Intel Macs supported (uses CPU/GPU fallback)
   - Dependency: `coremltools>=7.0` (installed with `[optional]` extras)

#### PyTorch Backend (Linux/Windows)

Located in `upscale.py`. Uses PyTorch-based Real-ESRGAN for 2x/4x image upscaling.

**Key features:**
- CUDA GPU acceleration (NVIDIA)
- CPU fallback for systems without GPU
- Models: `RealESRGAN_x2plus.pth` (2x), `realesr-general-x4v3.pth` (4x)
- Three-tier retry strategy for memory errors

#### Shared Mechanisms (All Backends)

1. **Marker System**: Creates `<image>.upscaled` files containing:
   - Scale factor
   - Model name
   - Device (coreml-ane/cuda/mps/cpu)
   - SHA256 hash of upscaled image
   - Source hash (original image hash for validation)

2. **Hash Verification**: On subsequent runs, compares SHA256 hashes to detect modified images and re-upscale only if needed.

3. **Graceful Shutdown**: Uses `threading.Event` (`_shutdown_event`) to handle Ctrl+C, cancelling pending operations without file corruption.

### Download Flow with Hash Verification

In `format/base.py`, the `save_chapter_images()` method:
1. Downloads images via `ChapterPageDownloader`
2. Checks for `.upscaled` marker and verifies hash (if upscale enabled)
3. If marker valid, skips server hash check
4. Otherwise, verifies against MangaDex-provided SHA256
5. Calls `Upscaler.process_images()` if enabled
6. Returns list of image paths for format conversion

### Network Layer (network.py)

**`requestsMangaDexSession`:**
- Custom `requests.Session` subclass
- Automatic rate limiting (handles 429 responses)
- Token refresh for OAuth2
- Retry logic (default: 5 attempts)

**`QueueWorker`:**
- Background thread processing `queue.Queue`
- Used for non-blocking download reports to MangaDex
- Graceful shutdown with `threading.Event`

## File Structure Reference

**Core Components:**
- `mangadex_downloader/__main__.py`: Package entry point
- `mangadex_downloader/main.py`: Main orchestration logic
- `mangadex_downloader/downloader.py`: `FileDownloader`, `ChapterPageDownloader`
- `mangadex_downloader/network.py`: HTTP session, rate limiting, `QueueWorker`
- `mangadex_downloader/upscale.py`: Real-ESRGAN PyTorch backend (Linux/Windows)
- `mangadex_downloader/upscale_coreml.py`: Real-ESRGAN Core ML backend (macOS)
- `mangadex_downloader/fetcher.py`: MangaDex API wrapper

**Format System:**
- `format/base.py`: Abstract base classes defining the pipeline
- `format/comic_book.py`: CBZ/CB7 implementation
- `format/pdf.py`: PDF implementation
- `format/epub.py`: EPUB implementation
- `format/utils.py`: Shared utilities (hash verification, etc.)

**Configuration:**
- `config/config.py`: `_Config` class and `ConfigProxy`
- `config/env.py`: Environment variable handling
- `config/utils.py`: Validation functions

**CLI:**
- `cli/download.py`: Main download command
- `cli/command.py`: Command routing

## Common Development Tasks

### Adding New Configuration Options
1. Add option to `_Config.confs` dictionary in `config/config.py` with `(default_value, validator_function)` tuple
2. Option automatically appears in `_Config.default_conf`
3. Add CLI argument in `cli/download.py` or appropriate CLI module
4. Access via `config.option_name` in application code

### Adding New Download Formats
1. Create new file in `format/` directory (e.g., `new_format.py`)
2. Inherit from appropriate base: `ConvertedChaptersFormat`, `ConvertedVolumesFormat`, or `ConvertedSingleFormat`
3. Implement required methods: `save()`, format-specific properties like `file_ext`
4. Register in `format/__init__.py` in the `formats` dictionary
5. Update `validate_format()` in `config/utils.py`

### Modifying Download Pipeline
The pipeline is in `BaseFormat.save_chapter_images()` in `format/base.py`:
1. Downloads images via `ChapterPageDownloader`
2. Verifies hashes (checks `.upscaled` marker first if upscaling enabled)
3. Calls `Upscaler.process_images()` if configured
4. Returns image paths

To add processing steps, modify this method or override in format subclass.

## Development Notes

### Python Version Requirements
- Minimum: Python 3.11
- Maximum: Python 3.12
- Development target: Python 3.10+ (as per ruff.toml)

### Known Circular Dependencies
These modules have circular dependencies resolved via lazy imports:
- `downloader.py` ↔ `config/config.py`
- `upscale.py` ↔ `config/config.py`
- `upscale_coreml.py` ↔ `config/config.py`
- `network.py` ↔ various modules

**Solution**: Always use lazy imports (import inside functions) when adding cross-module references.

### Memory Error Detection
The `Upscaler._is_memory_error()` method detects OOM by checking for these strings in exception messages:
- "out of memory"
- "cuda out of memory"
- "mps backend out of memory"

### Known Issues

**Circular Imports:**
- Use lazy imports (inside functions/methods) to break cycles
- Never import `_conf` at module level in `downloader.py`, `upscale.py`, or `network.py`

**Package Installation:**
- After ANY code change, run `uv pip install --upgrade ".[optional]"`
- Python caches modules, so reinstallation is mandatory for changes to take effect

### Development Best Practices

**CRITICAL - Always reinstall after code changes:**
```bash
uv pip install --upgrade ".[optional]"
```
This cannot be emphasized enough. Python caches modules and changes won't take effect without reinstallation.

**Project-specific practices:**
1. **Test with real MangaDex URLs** - Use actual manga chapters, not mocks
2. **Verify `.upscaled` markers** - After upscaling, check marker files contain correct hash and metadata
3. **Use `--log-level DEBUG`** - Shows actual execution flow, API calls, and hash verification steps

### Testing Approach
No automated test suite. Testing is manual via CLI:
```bash
# Test basic download
mangadex-dl "URL" --log-level DEBUG

# Test upscaling
mangadex-dl "URL" --upscale --upscale-scale 2

# Test specific format
mangadex-dl "URL" --save-as cbz

# Isolate single chapter for testing
mangadex-dl "URL" --start-chapter 1 --end-chapter 1

# View current configuration
mangadex-dl-conf --view
```

### Recent Significant Changes

**Upscaling Feature (October 2024):**
- Added Real-ESRGAN integration with automatic model downloading
- Implemented SHA256-based marker system for skip verification
- Added graceful shutdown with `threading.Event`
- Implemented memory error retry strategy with CPU fallback
- Fixed circular imports with lazy import pattern

**Configuration additions:**
- `upscale` (bool): Enable/disable upscaling
- `upscale_scale` (int): Scaling factor (2 or 4)
- `upscale_concurrency` (int): Parallel worker count
