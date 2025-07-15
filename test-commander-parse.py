#!/usr/bin/env python3
"""
test-commander-parse.py - Simple file parser for AI responses

Usage:
    python3 test-commander-parse.py
"""

import os

def main():
    filename = "resp.log"
    
    if not os.path.exists(filename):
        print(f"❌ File {filename} not found")
        return
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        print("❌ File is empty")
        return
    
    i = 0
    # Skip first line if it's ```tool_code
    if lines[0].strip() == '```tool_code':
        print("✅ Skipping ```tool_code line")
        i = 1
    
    current_file = None
    file_handle = None
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if we're starting a new file
        if line.startswith('---') and line.endswith('---') and current_file is None:
            # Extract filespec by chopping off first 3 and last 3 characters
            filespec = line[3:-3]
            print(f"📄 Found file: {filespec}")
            
            # Create backup if file exists
            if os.path.exists(filespec):
                backup_name = f"{filespec}.backup"
                os.rename(filespec, backup_name)
                print(f"📁 Created backup: {backup_name}")
            
            # Skip the next line (```type)
            i += 1
            if i < len(lines):
                print(f"   Skipping: {lines[i].strip()}")
                i += 1
            
            # Open file for writing
            current_file = filespec
            file_handle = open(filespec, 'w')
            print(f"✍️  Writing: {filespec}")
            
        # Check if we're ending current file
        elif line == '```' and current_file is not None:
            # Look ahead to see if next line starts a new file (---filespec---)
            if i + 1 < len(lines) and lines[i + 1].strip().startswith('---') and lines[i + 1].strip().endswith('---'):
                # This ``` ends the current file
                file_handle.close()
                print(f"✅ Completed: {current_file}")
                current_file = None
                file_handle = None
                i += 1
                continue
            else:
                # This ``` is part of the file content (like in markdown), write it
                file_handle.write(lines[i])
        
        # Write line to current file
        elif current_file is not None:
            file_handle.write(lines[i])
        
        i += 1
    
    # Close any remaining open file
    if file_handle is not None:
        file_handle.close()
        print(f"✅ Completed: {current_file} (EOF)")

if __name__ == "__main__":
    main()

