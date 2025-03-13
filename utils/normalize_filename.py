<<<<<<< HEAD
import os
import re

def normalize_filename(filename: str) -> str:
    """
    Normalize a file name by removing unsafe characters and normalizing whitespace,
    while preserving the file extension and handling edge cases.

    Rules:
    - Leading and trailing whitespace is removed.
    - For non-hidden files:
         * All dots ('.') in the base name are replaced with underscores.
         * Characters that are not alphanumeric, underscore, hyphen, or whitespace are removed.
         * Whitespace is collapsed into a single underscore.
    - For hidden files (those starting with a dot and having no extension):
         * The leading dot is preserved.
         * The rest of the filename is processed similarly.
    - The file extension is preserved (and converted to lowercase).
    - If the normalized base name ends up empty, it defaults to "file".
    """
    if not isinstance(filename, str):
        raise TypeError("Filename must be a string.")
    
    # Remove leading and trailing whitespace.
    filename = filename.strip()
    if not filename:
        raise ValueError("Filename cannot be empty or whitespace only.")
    
    # Split into base and extension.
    base, ext = os.path.splitext(filename)
    
    # Special handling for hidden files (e.g. ".env" or ".gitignore").
    if base.startswith('.') and ext == '':
        # Preserve the leading dot and replace any other dots in the rest.
        inner = base[1:].replace('.', '_')
        safe_inner = re.sub(r'[^\w\s-]', '', inner)
        safe_base = '.' + re.sub(r'\s+', '_', safe_inner)
    else:
        # Replace any dots in the base with underscores.
        base = base.replace('.', '_')
        safe_base = re.sub(r'[^\w\s-]', '', base)
        safe_base = re.sub(r'\s+', '_', safe_base)
    
    # If the safe base name is empty (or just a dot), use a default.
    if safe_base in ["", "."]:
        safe_base = "file"
    
    safe_ext = ext.lower()  # Normalize extension to lowercase.
    
    return safe_base + safe_ext

# --- Test Cases ---
if __name__ == "__main__":
    tests = [
        # 1. Basic filename with spaces and punctuation.
        ("GvHD patent background disease + target paper (2).pdf",
         "GvHD_patent_background_disease_target_paper_2.pdf"),
        # 2. Filename with leading and trailing whitespace.
        ("   my   file   name.txt  ", "my_file_name.txt"),
        # 3. Filename with unsafe characters.
        ("inv@lid*file:name?.doc", "invlidfilename.doc"),
        # 4. Filename with no extension.
        ("example file", "example_file"),
        # 5. Filename that is only whitespace -> should raise ValueError.
        ("   ", None),
        # 6. Filename with allowed underscore and hyphen.
        ("file_name-test.pdf", "file_name-test.pdf"),
        # 7. Filename with multiple dots (e.g., tar.gz).
        ("archive.backup.tar.gz", "archive_backup_tar.gz"),
        # 8. Filename with Unicode characters.
        ("résumé.doc", "résumé.doc"),
        # 9. Hidden file (starts with dot, no extension).
        (".env", ".env"),
        # 10. Filename that results in an empty base after cleaning.
        ("!!!.txt", "file.txt"),
        # 11. Filename with a mix of allowed and disallowed characters.
        ("This is a Test!@#$%^&*()_+-=[]{};'.pdf", "This_is_a_Test_-.pdf"),
        # 12. Filename with leading and trailing unsafe characters.
        ("###example###.mp3", "example.mp3"),
        # 13. Filename with a tab character.
        ("file\tname.pdf", "file_name.pdf"),
        # 14. Filename with a newline character.
        ("file\nname.doc", "file_name.doc"),
        # 15. Filename containing a forward slash.
        ("folder/file.txt", "folderfile.txt"),
        # 16. Filename containing a backslash.
        ("folder\\file.txt", "folderfile.txt"),
        # 17. Filename with mixed whitespace characters.
        ("file \r\n name.docx", "file_name.docx"),
        # 18. Filename with non-Latin characters and a space.
        ("文件 名称.pdf", "文件_名称.pdf"),
        # 19. Numeric filename.
        ("1234567890.txt", "1234567890.txt"),
        # 20. Hidden file with no extension (gitignore).
        (".gitignore", ".gitignore"),
    ]
    
    passed = 0
    failed = 0
    
    for idx, (input_filename, expected) in enumerate(tests, 1):
        try:
            result = normalize_filename(input_filename)
            if result == expected:
                print(f"Test case {idx} passed: '{input_filename}' -> '{result}'")
                passed += 1
            else:
                print(f"Test case {idx} FAILED: '{input_filename}' -> '{result}', expected '{expected}'")
                failed += 1
        except Exception as e:
            if expected is None:
                print(f"Test case {idx} passed (expected exception): '{input_filename}' raised '{e}'")
                passed += 1
            else:
                print(f"Test case {idx} FAILED: '{input_filename}' raised unexpected exception '{e}'")
                failed += 1
    
=======
import os
import re

def normalize_filename(filename: str) -> str:
    """
    Normalize a file name by removing unsafe characters and normalizing whitespace,
    while preserving the file extension and handling edge cases.

    Rules:
    - Leading and trailing whitespace is removed.
    - For non-hidden files:
         * All dots ('.') in the base name are replaced with underscores.
         * Characters that are not alphanumeric, underscore, hyphen, or whitespace are removed.
         * Whitespace is collapsed into a single underscore.
    - For hidden files (those starting with a dot and having no extension):
         * The leading dot is preserved.
         * The rest of the filename is processed similarly.
    - The file extension is preserved (and converted to lowercase).
    - If the normalized base name ends up empty, it defaults to "file".
    """
    if not isinstance(filename, str):
        raise TypeError("Filename must be a string.")
    
    # Remove leading and trailing whitespace.
    filename = filename.strip()
    if not filename:
        raise ValueError("Filename cannot be empty or whitespace only.")
    
    # Split into base and extension.
    base, ext = os.path.splitext(filename)
    
    # Special handling for hidden files (e.g. ".env" or ".gitignore").
    if base.startswith('.') and ext == '':
        # Preserve the leading dot and replace any other dots in the rest.
        inner = base[1:].replace('.', '_')
        safe_inner = re.sub(r'[^\w\s-]', '', inner)
        safe_base = '.' + re.sub(r'\s+', '_', safe_inner)
    else:
        # Replace any dots in the base with underscores.
        base = base.replace('.', '_')
        safe_base = re.sub(r'[^\w\s-]', '', base)
        safe_base = re.sub(r'\s+', '_', safe_base)
    
    # If the safe base name is empty (or just a dot), use a default.
    if safe_base in ["", "."]:
        safe_base = "file"
    
    safe_ext = ext.lower()  # Normalize extension to lowercase.
    
    return safe_base + safe_ext

# --- Test Cases ---
if __name__ == "__main__":
    tests = [
        # 1. Basic filename with spaces and punctuation.
        ("GvHD patent background disease + target paper (2).pdf",
         "GvHD_patent_background_disease_target_paper_2.pdf"),
        # 2. Filename with leading and trailing whitespace.
        ("   my   file   name.txt  ", "my_file_name.txt"),
        # 3. Filename with unsafe characters.
        ("inv@lid*file:name?.doc", "invlidfilename.doc"),
        # 4. Filename with no extension.
        ("example file", "example_file"),
        # 5. Filename that is only whitespace -> should raise ValueError.
        ("   ", None),
        # 6. Filename with allowed underscore and hyphen.
        ("file_name-test.pdf", "file_name-test.pdf"),
        # 7. Filename with multiple dots (e.g., tar.gz).
        ("archive.backup.tar.gz", "archive_backup_tar.gz"),
        # 8. Filename with Unicode characters.
        ("résumé.doc", "résumé.doc"),
        # 9. Hidden file (starts with dot, no extension).
        (".env", ".env"),
        # 10. Filename that results in an empty base after cleaning.
        ("!!!.txt", "file.txt"),
        # 11. Filename with a mix of allowed and disallowed characters.
        ("This is a Test!@#$%^&*()_+-=[]{};'.pdf", "This_is_a_Test_-.pdf"),
        # 12. Filename with leading and trailing unsafe characters.
        ("###example###.mp3", "example.mp3"),
        # 13. Filename with a tab character.
        ("file\tname.pdf", "file_name.pdf"),
        # 14. Filename with a newline character.
        ("file\nname.doc", "file_name.doc"),
        # 15. Filename containing a forward slash.
        ("folder/file.txt", "folderfile.txt"),
        # 16. Filename containing a backslash.
        ("folder\\file.txt", "folderfile.txt"),
        # 17. Filename with mixed whitespace characters.
        ("file \r\n name.docx", "file_name.docx"),
        # 18. Filename with non-Latin characters and a space.
        ("文件 名称.pdf", "文件_名称.pdf"),
        # 19. Numeric filename.
        ("1234567890.txt", "1234567890.txt"),
        # 20. Hidden file with no extension (gitignore).
        (".gitignore", ".gitignore"),
    ]
    
    passed = 0
    failed = 0
    
    for idx, (input_filename, expected) in enumerate(tests, 1):
        try:
            result = normalize_filename(input_filename)
            if result == expected:
                print(f"Test case {idx} passed: '{input_filename}' -> '{result}'")
                passed += 1
            else:
                print(f"Test case {idx} FAILED: '{input_filename}' -> '{result}', expected '{expected}'")
                failed += 1
        except Exception as e:
            if expected is None:
                print(f"Test case {idx} passed (expected exception): '{input_filename}' raised '{e}'")
                passed += 1
            else:
                print(f"Test case {idx} FAILED: '{input_filename}' raised unexpected exception '{e}'")
                failed += 1
    
>>>>>>> 5b6e3e1f6bb904635df1f05e870b8aeeed94cf1b
    print(f"\nTotal passed: {passed}, Total failed: {failed}")