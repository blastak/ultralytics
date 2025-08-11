#!/usr/bin/env python3
import os
import re
import csv

def count_words_in_file(filepath, word):
    """Count occurrences of a word in a file (case-insensitive)"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            return len(re.findall(word, content, re.IGNORECASE))
    except:
        return 0

def main():
    results = []
    total_obb = 0
    total_qbb = 0
    
    # Find all Python files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                obb_count = count_words_in_file(filepath, r'\bobb\b')
                qbb_count = count_words_in_file(filepath, r'\bqbb\b')
                difference = obb_count - qbb_count
                
                if obb_count > 0 or qbb_count > 0:  # Only include files with OBB or QBB
                    results.append([filepath, obb_count, qbb_count, difference])
                    total_obb += obb_count
                    total_qbb += qbb_count
    
    # Sort by absolute difference (largest first)
    results.sort(key=lambda x: abs(x[3]), reverse=True)
    
    # Write to CSV
    with open('obb_qbb_count_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File', 'OBB_Count', 'QBB_Count', 'Difference'])
        
        for result in results:
            writer.writerow(result)
        
        writer.writerow(['TOTAL', total_obb, total_qbb, total_obb - total_qbb])
    
    print(f"Results saved to obb_qbb_count_results.csv")
    print(f"Total OBB: {total_obb}")
    print(f"Total QBB: {total_qbb}")
    print(f"Difference: {total_obb - total_qbb}")

if __name__ == "__main__":
    main()