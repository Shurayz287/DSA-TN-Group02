import psutil
import os
import time
from pathlib import Path
from colorama import Fore, Style, init

# Config output
init(autoreset=True)

def print_summary(groups, representatives, all_processed_files, image_folder, metrics=None, total_initial=None, after_md5=None, 
                  elapsed_time=None, mem_used=None, method_name=""):
    print("\n" + "="*70)
    print(Fore.CYAN + Style.BRIGHT + f"📊 DUPLICATE DETECTION SUMMARY ({method_name.upper()})".center(70))
    print("="*70)

    if total_initial is not None:
        print(f"{Fore.YELLOW}Total Initial Images: {Fore.WHITE}{total_initial}")
    if after_md5 is not None:
        removed = total_initial - after_md5 if total_initial is not None else "?"
        print(f"{Fore.YELLOW}Remaining After MD5 Filter: {Fore.WHITE}{after_md5} "
              f"({removed} removed)")

    print(f"{Fore.YELLOW}Total Clusters Found: {Fore.WHITE}{len(groups)}")
    print(f"{Fore.YELLOW}Total Representatives: {Fore.WHITE}{len(representatives)}")
    print(f"{Fore.YELLOW}Total Processed Images: {Fore.WHITE}{len(all_processed_files)}")
    
    cleaned_folder = Path(image_folder) / f"Cleaned_{method_name}" # <-- Sửa logic của tôi
    print(f"{Fore.YELLOW}Cleaned Folder: {Fore.WHITE}{cleaned_folder}")

    print("\n" + "="*70)
    print(Fore.CYAN + Style.BRIGHT + "📊 PERFORMANCE EVALUATION".center(70))
    print("="*70)

    if elapsed_time is not None:
        print(f"{Fore.GREEN}Processing Time: {Fore.WHITE}{elapsed_time:.2f} seconds")
    if mem_used is not None:
        print(f"{Fore.GREEN}Memory Used: {Fore.WHITE}{mem_used:.2f} MB")

    print("\n" + "="*70)
    print(Fore.CYAN + Style.BRIGHT + "📊 SOME CLUSTERS".center(70))
    print("="*70)
    for idx, group in enumerate(groups):
        if idx >= 5:
            break
        rep = representatives[idx]
        rep_name = rep.name if hasattr(rep, "name") else Path(rep).name
        print(f"{Fore.CYAN}Cluster {idx + 1:>3} {Fore.WHITE}| {len(group):>3} images | "
              f"Representative: {Fore.GREEN}{rep_name}")
    print("-"*70)

    if metrics:
        print("\n" + "="*70)
        print(Fore.CYAN + Style.BRIGHT + "📊 EVALUATION METRICS".center(70))
        print("="*70)
        print(f"{Fore.CYAN}Precision: {Fore.WHITE}{metrics.get('precision', 0)*100:.2f}%")
        print(f"{Fore.CYAN}Recall:    {Fore.WHITE}{metrics.get('recall', 0)*100:.2f}%")
        print(f"{Fore.CYAN}F1-Score:  {Fore.WHITE}{metrics.get('f1_score', 0)*100:.2f}%")
    print("="*70 + "\n")


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 ** 2)