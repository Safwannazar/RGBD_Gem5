from pathlib import Path
import os
from RGBD import ModifiedRGBDProcessor  # Assuming you rename your txt to Good_RGBD.py

def test_single_pair():
    # --- USER INPUT: Update image paths below ---
    left_image_path = "frameL/img- (1).jpg"   # Update with your actual image
    right_image_path = "frameR/img- (1).jpg"  # Update with your actual image
    output_dir = "test_results"              # Folder to save output

    # Check if files exist
    if not os.path.exists(left_image_path) or not os.path.exists(right_image_path):
        print("❌ Left or Right image not found. Please check your path.")
        return

    # Initialize processor
    processor = ModifiedRGBDProcessor()

    # Run processing
    result = processor.process_single_pair(
        left_path=left_image_path,
        right_path=right_image_path,
        output_dir=output_dir,
        image_index=1,
        pair_name="img-(1)"
    )

    # Optionally display results
    if result:
        processor.display_focused_results(result)
        print("✅ Single image pair processing complete.")
    else:
        print("❌ Failed to process the image pair.")

if __name__ == "__main__":
    test_single_pair()
