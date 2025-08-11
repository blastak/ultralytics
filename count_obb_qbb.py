#!/usr/bin/env python3
import os
import re
import csv

def count_words_in_file(filepath, word):
    """Count occurrences of a word in a file (case-insensitive) - using simple case-insensitive match like grep -o"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Use simple case-insensitive search like grep -o does
            pattern = word
            matches = re.findall(pattern, content, re.IGNORECASE)
            return len(matches)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0

def debug_file_count(filepath, word):
    """Debug function to show actual matches in a file"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            matches = []
            pattern = word
            for i, line in enumerate(lines, 1):
                found = re.findall(pattern, line, re.IGNORECASE)
                if found:
                    matches.append((i, line.strip(), len(found)))
            return matches
    except:
        return []

def main():
    results = []
    total_obb = 0
    total_qbb = 0
    
    # Find all Python files
    for root, dirs, files in os.walk('.'):
        # Skip certain directories to avoid noise
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                obb_count = count_words_in_file(filepath, 'obb')
                qbb_count = count_words_in_file(filepath, 'qbb')
                difference = obb_count - qbb_count
                
                if obb_count > 0 or qbb_count > 0:  # Only include files with OBB or QBB
                    results.append([filepath, obb_count, qbb_count, difference])
                    total_obb += obb_count
                    total_qbb += qbb_count
                    
                    # Debug specific file that should have 8,8,0
                    if 'data/augment.py' in filepath:
                        print(f"\nDEBUG AUGMENT.PY: {filepath}")
                        print(f"  Raw counts: OBB={obb_count}, QBB={qbb_count}")
                        
                        # Test with bash grep to compare
                        import subprocess
                        try:
                            bash_obb = subprocess.run(['grep', '-i', '-o', 'obb', filepath], 
                                                    capture_output=True, text=True)
                            bash_qbb = subprocess.run(['grep', '-i', '-o', 'qbb', filepath], 
                                                    capture_output=True, text=True)
                            bash_obb_count = len(bash_obb.stdout.strip().split('\n')) if bash_obb.stdout.strip() else 0
                            bash_qbb_count = len(bash_qbb.stdout.strip().split('\n')) if bash_qbb.stdout.strip() else 0
                            print(f"  Bash grep counts: OBB={bash_obb_count}, QBB={bash_qbb_count}")
                        except:
                            pass
                        
                        obb_matches = debug_file_count(filepath, 'obb')
                        qbb_matches = debug_file_count(filepath, 'qbb')
                        print(f"  OBB matches ({len(obb_matches)}):")
                        for match in obb_matches:
                            print(f"    Line {match[0]}: {match[1]}")
                        print(f"  QBB matches ({len(qbb_matches)}):")
                        for match in qbb_matches:
                            print(f"    Line {match[0]}: {match[1]}")
                    
                    # Debug output for files with significant differences
                    elif abs(difference) > 3:
                        print(f"\nDEBUG: {filepath} (diff: {difference})")
                        if obb_count > qbb_count:
                            obb_matches = debug_file_count(filepath, 'obb')
                            qbb_matches = debug_file_count(filepath, 'qbb')
                            print(f"  OBB matches ({len(obb_matches)}): {obb_matches[:3]}")  # Show first 3
                            print(f"  QBB matches ({len(qbb_matches)}): {qbb_matches[:3]}")  # Show first 3
    
    # Sort by absolute difference (largest first)
    results.sort(key=lambda x: abs(x[3]), reverse=True)
    
    # Write to CSV
    with open('obb_qbb_count_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File', 'OBB_Count', 'QBB_Count', 'Difference'])
        
        for result in results:
            writer.writerow(result)
        
        writer.writerow(['TOTAL', total_obb, total_qbb, total_obb - total_qbb])
    
    print(f"\nResults saved to obb_qbb_count_results.csv")
    print(f"Total OBB: {total_obb}")
    print(f"Total QBB: {total_qbb}")
    print(f"Difference: {total_obb - total_qbb}")
    
    # Show top files with differences
    print(f"\nTop files with differences:")
    for result in results[:10]:
        if abs(result[3]) > 0:
            print(f"  {result[0]}: OBB={result[1]}, QBB={result[2]}, diff={result[3]}")

if __name__ == "__main__":
    main()