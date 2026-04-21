import os
import ast


def merge_match_files():
    # Initialize the merge dictionary and current index
    merged_dict = {}
    current_idx = 0
    output_file = "merged_matches.txt"

    i = 0
    while True:
        # Build the file path
        filename = os.path.join("match", f"match_{i:03d}.txt")

        # Check if the file exists
        if not os.path.exists(filename):
            break

        try:
            # Read the file content
            with open(filename, 'r') as f:
                content = f.read().strip()

            # Convert the string to a dictionary
            data = ast.literal_eval(content)

            # Sort by keys and extract values
            sorted_items = sorted(data.items(), key=lambda x: x[0])
            values = [item[1] for item in sorted_items]

            # Add values to the merge dictionary
            for value in values:
                merged_dict[current_idx] = value
                current_idx += 1

            #print(f"Processed {filename} with {len(values)} items")
            i += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            break

    # Save the merged results
    if merged_dict:
        with open(output_file, 'w') as f:
            f.write(str(merged_dict))
        print(f"\nSuccess! Merged results saved to {output_file}")
        print(f"Total items: {len(merged_dict)}")
    else:
        print("No valid data found to merge")


if __name__ == "__main__":
    merge_match_files()