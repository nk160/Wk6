import os
from pathlib import Path
import time
from datetime import datetime
import subprocess
import psutil
import json
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ArtworkDownloadHandler(FileSystemEventHandler):
    def __init__(self):
        self.start_time = time.time()
        self.file_counts = {
            'metadata': 0,
            'images': 0,
            'monet': 0,
            'vangogh': 0
        }
        self.last_file = None
        self.last_activity = time.time()
        
    def on_created(self, event):
        if event.is_directory:
            return
            
        path = Path(event.src_path)
        self.last_activity = time.time()
        self.last_file = path
        
        # Count files in specific directories
        if path.suffix.lower() == '.json':
            self.file_counts['metadata'] += 1
        elif path.suffix.lower() == '.jpg':
            self.file_counts['images'] += 1
            
        # Count paintings in final directories
        if path.parent.name == "monet":
            self.file_counts['monet'] += 1
        elif path.parent.name == "vangogh":
            self.file_counts['vangogh'] += 1

class VerboseMonitor:
    def __init__(self):
        self.data_dir = Path('/root/Wk6/data')
        self.wikiart_dir = Path('/root/Wk6/wikiart')
        self.monet_dir = self.data_dir / 'monet'
        self.vangogh_dir = self.data_dir / 'vangogh'
        self.temp_dir = self.data_dir / 'wikiart-saved'
        
        self.event_handler = ArtworkDownloadHandler()
        self.observer = Observer()
        
        # Count existing files on startup
        self.count_existing_files()
        
    def count_existing_files(self):
        """Count existing files in directories"""
        # Count Monet paintings
        if self.monet_dir.exists():
            self.event_handler.file_counts['monet'] = len(list(self.monet_dir.glob('*.jpg')))
            
        # Count Van Gogh paintings
        if self.vangogh_dir.exists():
            self.event_handler.file_counts['vangogh'] = len(list(self.vangogh_dir.glob('*.jpg')))
            
        # Count total images
        if self.temp_dir.exists():
            self.event_handler.file_counts['images'] = len(list(self.temp_dir.rglob('*.jpg')))
            
        # Count metadata files
        if self.temp_dir.exists():
            self.event_handler.file_counts['metadata'] = len(list(self.temp_dir.rglob('*.json')))

    def get_process_info(self):
        """Get detailed process information"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
            try:
                if 'wikiart.py' in ' '.join(proc.info['cmdline'] or []):
                    return {
                        'pid': proc.pid,
                        'cpu': proc.cpu_percent(),
                        'memory': proc.memory_info().rss / 1024 / 1024,  # MB
                        'running_time': time.time() - proc.create_time()
                    }
            except:
                continue
        return None

    def get_log_tail(self, lines=5):
        """Get the last few lines of the log file"""
        try:
            # Use absolute path and check both possible locations
            log_paths = [
                Path('/root/Wk6/wikiart/wikiart.py.log'),
                Path('wikiart/wikiart.py.log')
            ]
            
            for log_path in log_paths:
                if log_path.exists():
                    with open(log_path, 'r') as f:
                        # Read all lines and get the most recent ones
                        all_lines = f.readlines()
                        # Filter out empty lines and get last 'lines' number
                        recent_lines = [line for line in all_lines if line.strip()][-lines:]
                        return recent_lines
                        
            return ["No log file found"]
        except Exception as e:
            return [f"Error reading log: {e}"]

    def monitor(self):
        # Start file system observer
        self.observer.schedule(self.event_handler, str(self.data_dir), recursive=True)
        self.observer.start()
        
        try:
            while True:
                os.system('clear')
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Header
                print(f"\n🔍 Verbose Download Monitor - {current_time}")
                print("=" * 80)
                
                # Process Status
                proc_info = self.get_process_info()
                print("\n📊 PROCESS INFORMATION:")
                print("-" * 40)
                if proc_info:
                    print(f"Status: 🟢 ACTIVE")
                    print(f"PID: {proc_info['pid']}")
                    print(f"CPU Usage: {proc_info['cpu']:.1f}%")
                    print(f"Memory: {proc_info['memory']:.1f} MB")
                    print(f"Running Time: {proc_info['running_time']/60:.1f} minutes")
                else:
                    print("Status: ⭕ NO ACTIVE PROCESS")
                
                # File System Activity
                print("\n📁 FILE SYSTEM ACTIVITY:")
                print("-" * 40)
                print(f"Metadata Files: {self.event_handler.file_counts['metadata']}")
                print(f"Total Images: {self.event_handler.file_counts['images']}")
                print(f"Monet Paintings: {self.event_handler.file_counts['monet']}")
                print(f"Van Gogh Paintings: {self.event_handler.file_counts['vangogh']}")
                
                if self.event_handler.last_file:
                    print(f"\nLast File: {os.path.basename(self.event_handler.last_file)}")
                    print(f"Last Activity: {(time.time() - self.event_handler.last_activity):.1f} seconds ago")
                
                # Performance Metrics
                elapsed = time.time() - self.event_handler.start_time
                if self.event_handler.file_counts['images'] > 0:
                    rate = self.event_handler.file_counts['images'] / (elapsed/60)
                    print(f"\n⚡ PERFORMANCE METRICS:")
                    print("-" * 40)
                    print(f"Download Rate: {rate:.1f} images/minute")
                    print(f"Elapsed Time: {elapsed/60:.1f} minutes")
                
                print("\n(Press Ctrl+C to stop monitoring)")
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.observer.stop()
            print("\nMonitoring stopped by user")
        
        self.observer.join()

if __name__ == "__main__":
    # Install required package if not present
    try:
        import watchdog
    except ImportError:
        print("Installing required package: watchdog")
        subprocess.run(["pip", "install", "watchdog"])
    
    monitor = VerboseMonitor()
    monitor.monitor() 