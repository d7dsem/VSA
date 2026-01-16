# VSA-file

VSA-file is a specialized tool for spectral analysis and visualization of IQ recordings. It is a command-line tool that uses `matplotlib` and `numpy` to perform and visualize the analysis.

## Features

*   Spectral analysis of IQ data from `.bin` and `.wav` files.
*   Interactive visualization of the spectrum using `matplotlib`.
*   Support for different data types (currently `int16`).
*   Configurable FFT size, sample rate, and center frequency.
*   Ability to specify a starting sample offset.
*   Support for dBFS and linear amplitude scaling.
*   Smoothing of the spectrum using a Gaussian filter.
*   Ability to split large WAV files into smaller chunks.

## Installation

1. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main entry point is `vsa_file_entry.py`. You can run it from the command line with the following arguments:

```bash
python src/vsa_file_entry.py -i "path/to/<signal>.bin|wav" [--samp-rate <freq_hz>] [--samp-offs <Default:0>] [--dtype <Default:int16>] [--center-freq <freq_hz>] [--fft-n <Default:4096>] [--verbose]
```

### Arguments

*   `-i, --input`: Path to the IQ file (`.bin` or `.wav`). (Required)
*   `--samp-rate`: Sample rate in Hz. Required for `.bin` files. For `.wav` files, it's read from the header, and if provided, it must match.
*   `--samp-offs`: Sample offset in IQ samples (I/Q pairs). Default is 0.
*   `--dtype`: Expected sample data type. Currently, only `int16` is supported.
*   `--center-freq`: Center frequency in Hz. Default is 0.
*   `--fft-n`: FFT size. Default is 4096.
*   `--verbose`: Verbose output.

### Examples

*   Analyze a `.bin` file:
    ```bash
    python src/vsa_file_entry.py -i /path/to/your/signal.bin --samp-rate 2048000 --fft-n 8192
    ```

*   Analyze a `.wav` file with a specific center frequency:
    ```bash
    python src/vsa_file_entry.py -i /path/to/your/signal.wav --center-freq 1000000
    ```

## File Types

*   **`.bin`**: Raw interleaved IQ data. The data type is assumed to be `int16`.
*   **`.wav`**: WAV files with uncompressed PCM data. The tool validates the WAV header to ensure it's compatible.

## Dependencies

*   [matplotlib](https://matplotlib.org/)
*   [numpy](https://numpy.org/)
*   [scipy](https://scipy.org/)