#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
import os

_F_BUF_SZ = 4 * 1024 * 1024  # 4 MB buffer


class ConstellationPlot:
    def __init__(self, filename: str | Path | None = None, chunk_len: int | None = None, history_size=10000):
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        # Перетворюємо на Path, щоб .name працював
        self.fpath = Path(filename) if filename else None
        self.chunk_len = chunk_len
        self.history_size = history_size 
        
        # Буфер для історії (накопичення точок)
        self.history = np.zeros(history_size, dtype=np.complex64)
        self.h_idx = 0
        self.h_full = False

        self.symbols_proc = 0
        self.total_symbols = 0
        self.paused = False
        self.closed = False
        
        self.line, = self.ax.plot([], [], 'o', markersize=2, alpha=0.6, color='#1f77b4')
        
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_aspect('equal')
        
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('close_event', self._on_close)

    def update(self, symbols, title: str | None = None):
        if symbols.size == 0 or self.closed:
            return
            
        self.symbols_proc += len(symbols)
        
        # ЛОГІКА ХВОСТА: заповнюємо буфер по колу
        n = len(symbols)
        if n > self.history_size: # якщо чанк більший за буфер
            self.history = symbols[-self.history_size:]
            self.h_full = True
        else:
            end = self.h_idx + n
            if end <= self.history_size:
                self.history[self.h_idx:end] = symbols
                self.h_idx = end
            else:
                # Перехлест через край буфера
                first_part = self.history_size - self.h_idx
                self.history[self.h_idx:] = symbols[:first_part]
                self.history[:n-first_part] = symbols[first_part:]
                self.h_idx = n - first_part
                self.h_full = True

        # Малюємо історію, а не тільки останній чанк
        data_to_plot = self.history if self.h_full else self.history[:self.h_idx]
        self.line.set_data(data_to_plot.real, data_to_plot.imag)
        
        # TITLE FIX
        if title is None:
            title = f'Processed: {self.symbols_proc:_}'
            if self.total_symbols:
                title += f" / {self.total_symbols:_} ({100.0 * self.symbols_proc/self.total_symbols:.1f}%)"
            if self.fpath:
                title += f"\nFile: {self.fpath.name}" # property, not method
        
        self.ax.set_title(title)
        self.fig.canvas.draw_idle() 
        self.fig.canvas.flush_events()
        
    def _on_key(self, event):
        if event.key == 'escape':
            self.closed = True
            plt.close(self.fig)
        elif event.key == ' ':
            self.paused = not self.paused
            status = "PAUSED" if self.paused else "RUNNING"
            print(f"\n{status}")
            
    def _on_close(self, event):
        self.closed = True
        
        
    def is_closed(self):
        return self.closed
        
    def is_paused(self):
        return self.paused
        
    def show(self):
        """Show the plot"""
        self.fig.tight_layout()
        plt.show()
        
    def save(self, filename):
        """Save plot to file"""
        self.fig.tight_layout()
        self.fig.savefig(filename, dpi=150)


def process_file(fpath: Path|str, chunk_len: int, contsell_plot:ConstellationPlot):
    """Process file in chunks"""
    fpath = Path(fpath)
    if not fpath.exists():
        raise RuntimeError(f"Input file fakap! Check path! File: '{fpath}'.")
    file_size = fpath.stat().st_size

    ext = fpath.suffix
    match ext:
        case ".bin":
            symbol_size = 4  # sizeof(int16_t)*2
            dt = np.int16
            normalize = 32768.0
        case ".C16":
            symbol_size = 2  # sizeof(int8_t)*2
            dt = np.int8
            normalize = 128.0
        case ".dat":
            symbol_size = 8  # sizeof(liquid_float_complex)
            dt = np.float32
            normalize = None
        case _:
            raise RuntimeError(f"File '{fpath}' has suspicious extentsion. Suppostys IQ Interlived: .bin - int16, .C16 - int8 (HackRF), .dat - float32 (Liquid)")
    if file_size < chunk_len*symbol_size:
        print(f"File: '{fpath}' - too short ({file_size:_} B)")
        return
    
    total_symbols = file_size // symbol_size
    contsell_plot.total_symbols = total_symbols
    raw_data = np.empty(chunk_len*symbol_size, dtype=np.uint8)
    flat_float = np.empty(chunk_len*2, dtype=np.float32)
    f = open(fpath, 'rb', buffering=_F_BUF_SZ)
 
    symbols_read_total = 0
    # Enable interactive mode
    plt.ion()
    
    while symbols_read_total < total_symbols:
        # Check if closed
        if contsell_plot.is_closed():
            print("\nAborted by user (ESC)")
            break
            
        # Check if paused
        if contsell_plot.is_paused():
            plt.pause(0.1)
            continue
        # Read chunk
        bytes_read = f.readinto(raw_data)
        if bytes_read == 0:
            print(f"[EOF] breaking")
            break
        symbols_read = bytes_read // symbol_size
        # Convert to complex
        typed_data = raw_data[:symbols_read].view(dt)  # int16/int8/float32
        if normalize:
            flat_float[:len(typed_data)] = typed_data
            flat_float[:len(typed_data)] /= normalize
            symbols = flat_float[:len(typed_data)].view(np.complex64)
        else:
            symbols = typed_data.view(np.complex64)
        count = min(chunk_len, total_symbols - symbols_read)
        
        # Update plot
        contsell_plot.update(symbols)
        
        symbols_read_total += len(symbols)
        
        # Progress
        progress = (symbols_read_total * 100.0) / total_symbols
        print(f"Progress: {progress:5.1f}%\r", end='', flush=True)
        
        # Small pause for UI responsiveness
        plt.pause(0.001)
    
    f.close()
    
    # Turn off interactive mode
    plt.ioff()
    
    if not contsell_plot.is_closed():
        print(f"\nProcessed: {symbols_read_total} symbols")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize constellation from .dat file'
    )
    parser.add_argument('file', help='Input .dat file')
    parser.add_argument(
        '--chunk-len', 
        type=int, 
        default=1024, 
        help='Symbols per chunk (default: 1024)'
    )
    parser.add_argument(
        '-o', '--output', 
        help='Save to file instead of showing'
    )
    
    args = parser.parse_args()
    
    # Extract filename from path
    filename = os.path.basename(args.file)
    
    # Create plot
    plot = ConstellationPlot(filename, args.chunk_len)
    
    # Process file
    process_file(args.file, args.chunk_len, plot)
    
    # Show or save
    if args.output:
        plot.save(args.output)
        print(f"Saved to {args.output}")
    else:
        if not plot.is_closed():
            plot.show()


if __name__ == '__main__':
    main()