#!/usr/bin/env python3
"""
ZIP Extractor with Filename Shortening for Windows
Handles zip files with names too long for Windows filesystem

Usage: python zip_extractor.py <zip_file> [output_directory] [max_filename_length]
"""

import zipfile
import os
import sys
import re
import hashlib
from pathlib import Path
import unicodedata

def sanitize_filename(filename, max_length=100):
    """
    Sanitize and shorten filename while preserving extension
    """
    # Get the file extension
    name, ext = os.path.splitext(filename)

    # Remove or replace problematic characters
    # Remove unicode control characters and normalize
    name = unicodedata.normalize('NFKD', name)
    name = re.sub(r'[<>:"/\\|?*]', '_', name)  # Replace invalid Windows chars
    name = re.sub(r'\s+', '_', name)  # Replace multiple spaces with single underscore
    name = name.strip('._')  # Remove leading/trailing dots and underscores

    # Calculate available length for name (minus extension)
    available_length = max_length - len(ext)

    if len(name) > available_length:
        # Truncate and add hash to ensure uniqueness
        hash_part = hashlib.md5(filename.encode()).hexdigest()[:8]
        truncate_length = available_length - len(hash_part) - 1
        name = name[:truncate_length] + '_' + hash_part

    return name + ext

def extract_and_shorten_zip(zip_path, extract_to="extracted_files", max_filename_length=100):
    """
    Extract zip file and automatically shorten long filenames
    """
    # Create extraction directory
    extract_path = Path(extract_to)
    extract_path.mkdir(exist_ok=True)

    renamed_files = []
    extracted_count = 0

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            total_files = len([f for f in zip_ref.infolist() if not f.filename.endswith('/')])
            print(f"Extracting {total_files} files from {zip_path}...")

            for file_info in zip_ref.infolist():
                original_name = file_info.filename

                # Skip directories
                if original_name.endswith('/'):
                    continue

                # Get directory structure
                dir_parts = Path(original_name).parts[:-1]  # All parts except filename
                filename = Path(original_name).name

                # Shorten each directory part if needed
                shortened_dirs = []
                for part in dir_parts:
                    if len(part) > 50:  # Shorten directory names too
                        short_part = sanitize_filename(part, 50)
                        shortened_dirs.append(short_part)
                    else:
                        shortened_dirs.append(part)

                # Shorten filename if needed
                if len(filename) > max_filename_length:
                    short_filename = sanitize_filename(filename, max_filename_length)
                    renamed_files.append((original_name, short_filename))
                else:
                    short_filename = filename

                # Create full path with shortened components
                if shortened_dirs:
                    relative_path = Path(*shortened_dirs) / short_filename
                else:
                    relative_path = Path(short_filename)

                full_extract_path = extract_path / relative_path

                # Create directory structure
                full_extract_path.parent.mkdir(parents=True, exist_ok=True)

                # Extract file
                try:
                    with zip_ref.open(file_info) as source:
                        with open(full_extract_path, 'wb') as target:
                            target.write(source.read())
                    extracted_count += 1
                    if extracted_count % 10 == 0 or extracted_count == total_files:
                        print(f"Progress: {extracted_count}/{total_files} files extracted")

                except Exception as e:
                    print(f"✗ Failed to extract {original_name}: {e}")

            print(f"\n✓ Successfully extracted {extracted_count} files to: {extract_path.absolute()}")

            if renamed_files:
                print(f"\n📝 {len(renamed_files)} files were renamed to fit Windows limitations:")
                for original, shortened in renamed_files[:10]:  # Show first 10
                    print(f"  {Path(original).name} -> {shortened}")
                if len(renamed_files) > 10:
                    print(f"  ... and {len(renamed_files) - 10} more files")

    except zipfile.BadZipFile:
        print("❌ Error: Invalid or corrupted zip file")
        return False
    except FileNotFoundError:
        print(f"❌ Error: Zip file '{zip_path}' not found")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True

def main():
    """
    Command line interface
    """
    # if len(sys.argv) < 2:
    #     print("Usage: python zip_extractor.py <zip_file> [output_directory] [max_filename_length]")
    #     print("\nExample: python zip_extractor.py myfile.zip extracted_files 100")
    #     sys.exit(1)

    zip_file = "EmailsData.EMLextension.zip"
    output_dir = "emails"
    max_length = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    if not os.path.exists(zip_file):
        print(f"❌ Error: Zip file '{zip_file}' not found")
        sys.exit(1)

    print("🔧 ZIP File Extractor with Filename Shortening")
    print("=" * 50)
    print(f"Input file: {zip_file}")
    print(f"Output directory: {output_dir}")
    print(f"Max filename length: {max_length}")
    print()

    success = extract_and_shorten_zip(zip_file, output_dir, max_length)

    if success:
        print("\n✅ Extraction completed successfully!")
    else:
        print("\n❌ Extraction failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
