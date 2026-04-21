import os
import json
import ast


def clean_json_content(content):
    """Clean Markdown tags from the content"""
    content = content.strip()
    if "```" in content:
        try:
            start_marker = "```json" if "```json" in content else "```"
            start_index = content.find(start_marker) + len(start_marker)
            end_index = content.rfind("```")
            if end_index > start_index:
                content = content[start_index:end_index]
        except Exception:
            pass
    return content.strip()


def merge_gene_results():
    merged_data = {}
    current_idx = 0
    output_file = "merged_results.txt"
    log_file = "merge_error_log.txt"

    # List to record issues
    issues = []

    i = 0
    consecutive_missing = 0  # Used to determine if the sequence has truly ended

    print("Starting merge and inspection...")

    while True:
        # Allow for file name interruption attempts (to prevent stopping completely due to an occasional missing file)
        # However, we generally assume that if generated sequentially, encountering a non-existent file indicates the end.
        filename = os.path.join("result", f"num_{i:03d}_result.txt")

        if not os.path.exists(filename):
            # If the file does not exist, we assume the processing is finished.
            # Record the last attempted file name to help confirm if it was interrupted midway.
            print(f"Scanning complete. The last file looked for was: {filename} (not found)")
            break

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                raw_content = f.read()

            # 1. Check if the file is empty
            if not raw_content.strip():
                print(f"  [Warning] File is empty: {filename}")
                issues.append(f"File is empty: {filename}")
                i += 1
                continue

            # 2. Clean and parse
            json_str = clean_json_content(raw_content)
            data = {}
            parse_success = False

            try:
                data = json.loads(json_str)
                parse_success = True
            except json.JSONDecodeError:
                try:
                    data = ast.literal_eval(json_str)
                    parse_success = True
                except Exception:
                    pass

            # 3. Check parsing results
            if not parse_success or not isinstance(data, dict):
                print(f"  [Error] Parsing failed: {filename}")
                issues.append(f"JSON parsing failed: {filename}")
                i += 1
                continue

            # 4. Check the number of Spots (Core requirement)
            count = len(data)
            if count != 10:
                msg = f"Abnormal count (contains {count} spots): {filename}"
                # Note: It's normal for the last file to have fewer than 10 spots; we manually exclude this when reporting at the end.
                print(f"  [Info] {msg}")
                issues.append(msg)

            # 5. Merge data
            # Even if the count is incorrect, try to merge the existing data anyway.
            sorted_keys = sorted(data.keys(), key=lambda x: int(x))
            for key in sorted_keys:
                merged_data[str(current_idx)] = {
                    "predicted_cell_type": data[key]["predicted_cell_type"]
                }
                current_idx += 1

        except Exception as e:
            err_msg = f"Unknown error {filename}: {str(e)}"
            print(f"  {err_msg}")
            issues.append(err_msg)

        # Continue to the next one
        i += 1

    if issues:
        print("\n--- List of Abnormal Files ---")
        for issue in issues:
            print(f" -> {issue}")

        # Write exceptions to the log file
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("Merge Error Log\n")
            f.write("================\n")
            for issue in issues:
                f.write(issue + "\n")

    # Save merged data
    if merged_data:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        print(f"Merged results saved to: {output_file}")


if __name__ == "__main__":
    merge_gene_results()