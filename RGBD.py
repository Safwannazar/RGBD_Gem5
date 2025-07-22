# RGBD Deep Learning Pipeline for Stereo Bird's Eye Chili Images - IMPROVED ALIGNMENT VERSION
# Install required packages first:
# pip install opencv-python numpy matplotlib torch torchvision transformers pillow scikit-image

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import pipeline
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import imageio

class ModifiedRGBDProcessor:
    def __init__(self):
        """Initialize the RGBD processor with deep learning models"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        # Insert your calibration parameters here
        self.camera_matrix_left = np.array([
            [1587.5034, 0, 1035.2395],
            [0, 1586.8911, 486.2811],
            [0, 0, 1]
        ])

        self.camera_matrix_right = np.array([
            [1580.8109, 0, 977.9626],
            [0, 1582.3192, 486.7659],
            [0, 0, 1]
        ])

        self.dist_coeffs_left = np.array([-0.4003, 0.1690, 0, 0, -0.1365])
        self.dist_coeffs_right = np.array([-0.3833, 0.0756, 0, 0, 0.0496])

        self.R = cv2.Rodrigues(np.array([0.0007, 0.0061, -0.0117]))[0]
        self.T = np.array([-112.6864, -1.7989, -0.1721]) / 1000.0  # Convert mm to meters
        self.baseline = np.linalg.norm(self.T)
        
        # Initialize depth estimation model
        self.depth_estimator = None
        self.load_depth_model()
        
        # Initialize multiple stereo matchers with different parameters
        self.init_stereo_matchers()
        
    def load_depth_model(self):
        """Load pre-trained depth estimation model"""
        try:
            # Using MiDaS model for monocular depth estimation
            self.depth_estimator = pipeline("depth-estimation", 
                                           model="Intel/dpt-large",
                                           device=0 if torch.cuda.is_available() else -1,
                                           use_fast=True)
            print("✓ Depth estimation model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load depth model: {e}")
            print("Will use traditional stereo matching instead")
    
    def init_stereo_matchers(self):
        """Initialize multiple stereo matchers with different parameters"""
        # StereoBM with improved parameters
        self.stereo_bm = cv2.StereoBM_create()
        self.stereo_bm.setNumDisparities(16*12)  # Increased disparities
        self.stereo_bm.setBlockSize(21)  # Larger block size for better matching
        self.stereo_bm.setPreFilterCap(31)
        self.stereo_bm.setPreFilterSize(9)
        self.stereo_bm.setMinDisparity(0)
        self.stereo_bm.setTextureThreshold(10)
        self.stereo_bm.setUniquenessRatio(15)
        self.stereo_bm.setSpeckleWindowSize(100)
        self.stereo_bm.setSpeckleRange(32)
        
        # StereoSGBM for better quality
        self.stereo_sgbm = cv2.StereoSGBM_create()
        self.stereo_sgbm.setNumDisparities(16*12)
        self.stereo_sgbm.setBlockSize(11)
        self.stereo_sgbm.setP1(8*3*11**2)
        self.stereo_sgbm.setP2(32*3*11**2)
        self.stereo_sgbm.setMinDisparity(0)
        self.stereo_sgbm.setUniquenessRatio(10)
        self.stereo_sgbm.setSpeckleWindowSize(100)
        self.stereo_sgbm.setSpeckleRange(32)
        self.stereo_sgbm.setDisp12MaxDiff(1)
        self.stereo_sgbm.setMode(cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    
    def load_stereo_images(self, left_path, right_path):
        """Load and preprocess stereo image pair"""
        try:
            left_img = cv2.imread(left_path)
            right_img = cv2.imread(right_path)
            
            if left_img is None or right_img is None:
                raise ValueError("Could not load one or both images")
            
            # Convert to RGB for processing
            left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
            
            return left_rgb, right_rgb, left_img, right_img
        except Exception as e:
            print(f"Error loading images: {e}")
            return None, None, None, None
    
    def find_feature_matches(self, img1, img2):
        """Find feature matches between two images using SIFT"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return None, None, None, None
        
        # Match features using FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test to get good matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        return kp1, kp2, good_matches, (des1, des2)
    
    def advanced_stereo_rectification(self, left_img, right_img):
        """Advanced stereo rectification using feature matching and homography"""
        print("Applying advanced stereo rectification...")
        
        height, width = left_img.shape[:2]
        
        # Find feature matches
        kp1, kp2, good_matches, _ = self.find_feature_matches(left_img, right_img)
        
        if kp1 is None or len(good_matches) < 10:
            print("Insufficient feature matches, falling back to simple alignment")
            return self.simple_stereo_rectification_improved(left_img, right_img)
        
        print(f"Found {len(good_matches)} good feature matches")
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography with RANSAC
        try:
            H, mask = cv2.findHomography(dst_pts, src_pts, 
                                       cv2.RANSAC, 
                                       ransacReprojThreshold=3.0,
                                       maxIters=5000,
                                       confidence=0.995)
            
            if H is None:
                print("Homography estimation failed, using simple rectification")
                return self.simple_stereo_rectification_improved(left_img, right_img)
            
            # Apply homography to right image
            right_rectified = cv2.warpPerspective(right_img, H, (width, height))
            left_rectified = left_img.copy()
            
            print(f"✓ Advanced rectification applied with {np.sum(mask)} inliers")
            
        except Exception as e:
            print(f"Advanced rectification failed: {e}")
            return self.simple_stereo_rectification_improved(left_img, right_img)
        
        # Create anaglyph visualization
        anaglyph_result = self.create_anaglyph_stereo(left_rectified, right_rectified)
        
        return left_rectified, right_rectified, anaglyph_result
    
    def simple_stereo_rectification_improved(self, left_img, right_img):
        """Improved simple rectification with better alignment"""
        print("Applying improved simple stereo rectification...")
        
        height, width = left_img.shape[:2]
        
        # Convert to grayscale for alignment
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        # Enhanced horizontal alignment using multiple template matching regions
        best_offset = 0
        best_correlation = 0
        
        # Test multiple horizontal regions for robust alignment
        test_regions = [
            (height//4, height//4 + height//8),    # Upper region
            (height//2 - height//16, height//2 + height//16),  # Center region  
            (3*height//4 - height//8, 3*height//4)  # Lower region
        ]
        
        total_offset = 0
        valid_regions = 0
        
        for start_row, end_row in test_regions:
            left_roi = left_gray[start_row:end_row, :]
            right_roi = right_gray[start_row:end_row, :]
            
            # Template matching for horizontal alignment
            template_width = min(width//3, 200)  # Use 1/3 of image width or 200px max
            template = left_roi[:, width//4:width//4 + template_width]
            
            try:
                result = cv2.matchTemplate(right_roi, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > 0.3:  # Lower threshold for acceptance
                    region_offset = max_loc[0] - width//4
                    total_offset += region_offset
                    valid_regions += 1
                    print(f"Region {start_row}-{end_row}: offset={region_offset}, correlation={max_val:.3f}")
                
            except Exception as e:
                print(f"Template matching failed for region {start_row}-{end_row}: {e}")
                continue
        
        # Calculate average offset
        if valid_regions > 0:
            avg_offset = total_offset / valid_regions
            print(f"Average horizontal offset: {avg_offset:.1f} pixels from {valid_regions} regions")
            
            # Apply correction if offset is significant
            if abs(avg_offset) > 2:  # Lower threshold for correction
                M = np.float32([[1, 0, -avg_offset], [0, 1, 0]])
                right_rectified = cv2.warpAffine(right_img, M, (width, height))
                left_rectified = left_img.copy()
                print(f"✓ Applied horizontal correction: {-avg_offset:.1f} pixels")
            else:
                left_rectified = left_img.copy()
                right_rectified = right_img.copy()
                print("✓ Images already well aligned")
        else:
            print("No reliable alignment found, using original images")
            left_rectified = left_img.copy() 
            right_rectified = right_img.copy()
        
        # Create anaglyph visualization
        anaglyph_result = self.create_anaglyph_stereo(left_rectified, right_rectified)
        
        return left_rectified, right_rectified, anaglyph_result
    
    def create_anaglyph_stereo(self, left_img, right_img):
        """Create anaglyph (red-cyan) stereo image for rectification visualization"""
        print("Creating anaglyph stereo visualization...")
        
        # Convert to grayscale if needed
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img.copy()
            right_gray = right_img.copy()
        
        # Create anaglyph image
        # Red channel from left image, Green and Blue channels from right image
        anaglyph = np.zeros((left_gray.shape[0], left_gray.shape[1], 3), dtype=np.uint8)
        anaglyph[:, :, 0] = left_gray  # Red channel (left eye)
        anaglyph[:, :, 1] = right_gray  # Green channel (right eye) 
        anaglyph[:, :, 2] = right_gray  # Blue channel (right eye)
        
        return anaglyph
    
    def deep_learning_depth(self, image):
        """Deep learning-based monocular depth estimation"""
        if self.depth_estimator is None:
            print("Depth estimator not available")
            return None, None
        
        try:
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                image_pil = Image.fromarray(image)
            else:
                image_pil = image
            
            # Get depth estimation
            depth_result = self.depth_estimator(image_pil)
            depth_map = np.array(depth_result['depth'])
            
            # Normalize depth map
            depth_normalized = ((depth_map - depth_map.min()) / 
                              (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
            
            return depth_map, depth_normalized
        except Exception as e:
            print(f"Error in deep learning depth estimation: {e}")
            return None, None
    
    def overlay_depth_on_image(self, rgb_image, depth_map, alpha=0.6):
        """Overlay depth map on RGB image with improved handling"""
        if depth_map is None:
            print("Warning: depth_map is None, returning original image")
            return rgb_image
    
        # Ensure both images have the same dimensions
        if depth_map.shape[:2] != rgb_image.shape[:2]:
            print(f"Resizing depth map from {depth_map.shape[:2]} to {rgb_image.shape[:2]}")
            depth_map = cv2.resize(depth_map, (rgb_image.shape[1], rgb_image.shape[0]))
    
        # Convert depth to 3-channel colormap for overlay
        if len(depth_map.shape) == 2:
            # Apply colormap to single-channel depth
            depth_colored = cv2.applyColorMap(depth_map.astype(np.uint8), cv2.COLORMAP_JET)
            # Convert from BGR to RGB for consistency
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        else:
            depth_colored = depth_map
    
        # Ensure RGB image is in RGB format (not BGR)
        if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
            rgb_for_overlay = rgb_image.copy()
        else:
            rgb_for_overlay = rgb_image
    
        # Create overlay using weighted addition
        overlay = cv2.addWeighted(rgb_for_overlay.astype(np.uint8), 1-alpha, 
                                depth_colored.astype(np.uint8), alpha, 0)
    
        return overlay
    
    def create_stereo_vs_dl_comparison(self, stereo_depth, dl_depth):
        """Create side-by-side comparison of stereo vs DL depth with JET colormap"""
        if stereo_depth is None or dl_depth is None:
            return None
        
        # Ensure both have same dimensions
        if stereo_depth.shape != dl_depth.shape:
            stereo_depth = cv2.resize(stereo_depth, (dl_depth.shape[1], dl_depth.shape[0]))
        
        # Apply JET colormap to both depth maps
        stereo_colored = cv2.applyColorMap(stereo_depth.astype(np.uint8), cv2.COLORMAP_JET)
        dl_colored = cv2.applyColorMap(dl_depth.astype(np.uint8), cv2.COLORMAP_JET)
        
        # Convert from BGR to RGB for display
        stereo_colored = cv2.cvtColor(stereo_colored, cv2.COLOR_BGR2RGB)
        dl_colored = cv2.cvtColor(dl_colored, cv2.COLOR_BGR2RGB)
        
        # Create side-by-side comparison
        comparison = np.hstack([stereo_colored, dl_colored])
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Add white text with black outline for better visibility
        def add_text_with_outline(img, text, position, font, font_scale, thickness):
            # Black outline
            cv2.putText(img, text, position, font, font_scale, (0, 0, 0), thickness + 2)
            # White text
            cv2.putText(img, text, position, font, font_scale, (255, 255, 255), thickness)
        
        add_text_with_outline(comparison, 'STEREO DEPTH', (10, 30), font, font_scale, thickness)
        add_text_with_outline(comparison, 'DL DEPTH', (dl_depth.shape[1] + 10, 30), font, font_scale, thickness)
        
        return comparison
    
    def save_individual_results(self, results_dict, base_output_dir="output", image_index=1):
        """Save each result to its own folder with incremental naming"""
        
        # Create main output directory
        Path(base_output_dir).mkdir(exist_ok=True)
        
        # Define folder and file mappings with incremental numbering
        save_mapping = {
            'dl_depth': ('DLDepth', f'DLDepth{image_index}.jpg'),
            'depth_overlay': ('DLDepthOverlay', f'DLDepthOverlay{image_index}.jpg'),
            'stereo_vs_dl_comparison': ('StereoVsDL', f'StereoVsDL{image_index}.jpg'),
            'strong_depth_overlay': ('StrongDepthOverlay', f'StrongDepthOverlay{image_index}.jpg'),
            'anaglyph_rectified': ('AnaglyphRectified', f'AnaglyphRectified{image_index}.jpg'),
            'histogram_comparison': ('HistogramComparisons', f'DepthHistogram{image_index}.png')
        
        }
        
        saved_files = []
        
        for key, (folder_name, file_name) in save_mapping.items():
            if key in results_dict and results_dict[key] is not None:
                # Create subfolder
                folder_path = Path(base_output_dir) / folder_name
                folder_path.mkdir(exist_ok=True)
                
                # Full file path
                file_path = folder_path / file_name
                
                image = results_dict[key]
                
                # Convert RGB to BGR for OpenCV if needed (for color images)
                if len(image.shape) == 3 and image.shape[2] == 3:
                    if key in ['depth_overlay', 'strong_depth_overlay', 'stereo_vs_dl_comparison', 'anaglyph_rectified']:
                        # These are RGB format, convert to BGR for OpenCV
                        image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    else:
                        image_to_save = image
                else:
                    # Grayscale image
                    image_to_save = image
                
                # Save the image
                cv2.imwrite(str(file_path), image_to_save)
                saved_files.append(str(file_path))
                print(f"✓ Saved: {file_path}")
        
        return saved_files
    
    def robust_stereo_matching(self, left_img, right_img):
        """Robust stereo matching for comparison purposes"""
        print("Computing robust stereo matching...")
        
        # Convert to grayscale and preprocess
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        left_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(left_gray)
        right_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(right_gray)
        
        # Compute disparity using SGBM
        disparity = self.stereo_sgbm.compute(left_eq, right_eq)
        
        # Post-process
        disp_filtered = disparity.astype(np.float32) / 16.0
        disp_filtered[disp_filtered < 0] = 0
        
        # Normalize for visualization
        disp_normalized = cv2.normalize(disp_filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        return disp_filtered, disp_normalized

    def get_image_pairs_from_folders(self, left_folder, right_folder):
        """Get matching image pairs from left and right folders with proper numeric sorting"""
        left_path = Path(left_folder)
        right_path = Path(right_folder)
    
        if not left_path.exists() or not right_path.exists():
            print(f"❌ Folder not found: {left_folder} or {right_folder}")
            return []
    
        # Get all image files from both folders
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
        left_files = {}
        right_files = {}
    
        # Process left folder files
        for f in left_path.iterdir():
            if f.suffix.lower() in image_extensions:
                # Extract number from filename like "img- (1).jpg"
                filename = f.stem  # Gets filename without extension
                try:
                    # Extract number from pattern like "img- (1)"
                    if "img-" in filename and "(" in filename and ")" in filename:
                        number_part = filename.split("(")[1].split(")")[0].strip()
                        number = int(number_part)
                        left_files[number] = f
                    else:
                        print(f"⚠️ Skipping file with unexpected format: {f.name}")
                except (ValueError, IndexError):
                    print(f"⚠️ Could not extract number from: {f.name}")
    
        # Process right folder files  
        for f in right_path.iterdir():
            if f.suffix.lower() in image_extensions:
                filename = f.stem
                try:
                    if "img-" in filename and "(" in filename and ")" in filename:
                        number_part = filename.split("(")[1].split(")")[0].strip()
                        number = int(number_part)
                        right_files[number] = f
                    else:
                        print(f"⚠️ Skipping file with unexpected format: {f.name}")
                except (ValueError, IndexError):
                    print(f"⚠️ Could not extract number from: {f.name}")
    
    # Find matching pairs and sort by number (ascending order)
        matching_pairs = []
        common_numbers = set(left_files.keys()) & set(right_files.keys())
        
        for number in sorted(common_numbers):  # This ensures ascending order
            left_file = left_files[number]
            right_file = right_files[number]
            pair_name = f"img-({number})"
            matching_pairs.append((str(left_file), str(right_file), pair_name))
        
        # Show missing pairs
        left_only = set(left_files.keys()) - set(right_files.keys())
        right_only = set(right_files.keys()) - set(left_files.keys())
        
        if left_only:
            print(f"⚠️ Left images without matching right: {sorted(left_only)}")
        if right_only:
            print(f"⚠️ Right images without matching left: {sorted(right_only)}")
        
        print(f"✓ Found {len(matching_pairs)} matching image pairs in ascending order")
        for left_path, right_path, pair_name in matching_pairs:
            print(f"   {pair_name}: {Path(left_path).name} + {Path(right_path).name}")
        
        return matching_pairs

    def process_multiple_image_pairs(self, left_folder, right_folder, output_dir="batch_results"):
        """Process multiple image pairs from folders"""
        print("="*80)
        print("BATCH PROCESSING - Multiple Image Pairs")
        print("="*80)
        
        # Get image pairs
        image_pairs = self.get_image_pairs_from_folders(left_folder, right_folder)
        
        if not image_pairs:
            print("❌ No matching image pairs found!")
            return
        
        all_results = []
        successful_pairs = 0
        
        for i, (left_path, right_path, pair_name) in enumerate(image_pairs, 1):
            print(f"\n{'='*60}")
            print(f"Processing pair {i}/{len(image_pairs)}: {pair_name}")
            print(f"Left: {Path(left_path).name}")
            print(f"Right: {Path(right_path).name}")
            print(f"{'='*60}")
            
            try:
                # Process single image pair
                results = self.process_single_pair(left_path, right_path, output_dir, i, pair_name)
                
                if results:
                    all_results.append({
                        'pair_name': pair_name,
                        'index': i,
                        'results': results,
                        'left_path': left_path,
                        'right_path': right_path
                    })
                    successful_pairs += 1
                    print(f"✅ Successfully processed pair {i}: {pair_name}")
                else:
                    print(f"❌ Failed to process pair {i}: {pair_name}")
                    
            except Exception as e:
                print(f"❌ Error processing pair {i} ({pair_name}): {e}")
                continue
        
        # Summary
        print(f"\n{'='*80}")
        print(f"🎯 BATCH PROCESSING COMPLETED!")
        print(f"✅ Successfully processed: {successful_pairs}/{len(image_pairs)} pairs")
        print(f"📁 Results saved in: {output_dir}/")
        print(f"{'='*80}")
        
        # Show folder structure
        self.show_output_structure(output_dir, successful_pairs)
        
        return all_results

    def process_single_pair(self, left_path, right_path, output_dir, image_index, pair_name):
        """Process a single image pair"""
        
        # Load stereo images
        left_rgb, right_rgb, left_bgr, right_bgr = self.load_stereo_images(left_path, right_path)
        
        if left_rgb is None:
            print(f"❌ Failed to load images: {pair_name}")
            return None
        
        print(f"✓ Images loaded - Size: {left_rgb.shape}")
        
        # Advanced stereo rectification with better alignment
        print("→ Applying advanced stereo rectification...")
        left_rect, right_rect, anaglyph_rectified = self.advanced_stereo_rectification(left_bgr, right_bgr)
        
        # Deep learning depth estimation on LEFT image
        print("→ Computing deep learning depth estimation...")
        depth_dl_raw, depth_dl_normalized = self.deep_learning_depth(left_rgb)
        
        if depth_dl_raw is None:
            print("❌ Deep learning depth estimation failed")
            return None
        
        # Create DL depth overlay on LEFT image
        print("→ Creating DL depth overlay...")
        depth_overlay = self.overlay_depth_on_image(left_rgb, depth_dl_normalized, alpha=0.5)
        
        # Create strong depth overlay (α=0.8)
        print("→ Creating strong depth overlay...")
        strong_depth_overlay = self.overlay_depth_on_image(left_rgb, depth_dl_normalized, alpha=0.8)
        
        # Create stereo depth for comparison
        print("→ Computing stereo depth...")
        stereo_depth_raw, stereo_depth_normalized = self.robust_stereo_matching(left_rect, right_rect)
        
        # Create stereo vs DL comparison with JET colormap
        print("→ Creating stereo vs DL comparison...")
        stereo_vs_dl_comparison = self.create_stereo_vs_dl_comparison(stereo_depth_normalized, depth_dl_normalized)
        
        # NEW: Create depth histogram comparison
        print("→ Creating depth histogram comparison...")
        histogram_fig, histogram_stats = self.create_depth_histogram_comparison(
            stereo_depth_normalized, depth_dl_normalized, pair_name
        )
        
        # Save histogram plot
        if histogram_fig:
            histogram_path = Path(output_dir) / "HistogramComparisons"
            Path(output_dir).mkdir(exist_ok=True) 
            histogram_path.mkdir(exist_ok=True)
            histogram_save_path = histogram_path / f"DepthHistogram{image_index}.png"
            histogram_fig.savefig(str(histogram_save_path), dpi=150, bbox_inches='tight')
            plt.close(histogram_fig)  # Close to free memory
            print(f"✓ Histogram saved: {histogram_save_path}")
        
        # NEW: Analyze quality metrics
        quality_metrics = self.analyze_depth_quality_metrics(stereo_depth_normalized, depth_dl_normalized)
        
        # Prepare results dictionary
        results = {
            'dl_depth': depth_dl_normalized,
            'depth_overlay': depth_overlay,
            'stereo_vs_dl_comparison': stereo_vs_dl_comparison,
            'strong_depth_overlay': strong_depth_overlay,
            'anaglyph_rectified': anaglyph_rectified
        }
        
        # Save individual results with incremental naming
        print("→ Saving results...")
        saved_files = self.save_individual_results(results, output_dir, image_index)
        
        return results

    def show_output_structure(self, output_dir, num_pairs):
        """Show the output folder structure"""
        print(f"\n📂 Output Structure in '{output_dir}':")
        folders = ['DLDepth', 'DLDepthOverlay', 'StereoVsDL', 'StrongDepthOverlay', 'AnaglyphRectified', 'HistogramComparisons']
        
        for folder in folders:
            print(f"├── {folder}/")
            for i in range(1, num_pairs + 1):
                prefix = "│   ├──" if i < num_pairs else "│   └──"
                print(f"{prefix} {folder}{i}.jpg")
        
        print(f"\n🌈 All depth comparisons use JET colormap")
        print(f"🔴🔵 Anaglyph images show improved alignment")
        """Main processing function focused on DL depth results and improved anaglyph rectification"""
        print("="*80)
        print("IMPROVED RGBD Pipeline - Better Alignment + Jet Colormap")
        print("="*80)
        
        # Load stereo images
        print("1. Loading stereo images...")
        left_rgb, right_rgb, left_bgr, right_bgr = self.load_stereo_images(left_path, right_path)
        
        if left_rgb is None:
            print("❌ Failed to load images")
            return
        
        print(f"✓ Images loaded - Size: {left_rgb.shape}")
        
        # Advanced stereo rectification with better alignment
        print("2. Applying advanced stereo rectification...")
        left_rect, right_rect, anaglyph_rectified = self.advanced_stereo_rectification(left_bgr, right_bgr)
        
        # Deep learning depth estimation on LEFT image
        print("3. Computing deep learning depth estimation...")
        depth_dl_raw, depth_dl_normalized = self.deep_learning_depth(left_rgb)
        
        if depth_dl_raw is None:
            print("❌ Deep learning depth estimation failed")
            return
        
        # Create DL depth overlay on LEFT image
        print("4. Creating DL depth overlay on left image...")
        depth_overlay = self.overlay_depth_on_image(left_rgb, depth_dl_normalized, alpha=0.5)
        
        # Create strong depth overlay (α=0.8)
        print("5. Creating strong depth overlay...")
        strong_depth_overlay = self.overlay_depth_on_image(left_rgb, depth_dl_normalized, alpha=0.8)
        
        # Create stereo depth for comparison
        print("6. Computing stereo depth for comparison...")
        stereo_depth_raw, stereo_depth_normalized = self.robust_stereo_matching(left_rect, right_rect)
        
        # Create stereo vs DL comparison with JET colormap
        print("7. Creating stereo vs DL depth comparison with JET colormap...")
        stereo_vs_dl_comparison = self.create_stereo_vs_dl_comparison(stereo_depth_normalized, depth_dl_normalized)
        
        # Prepare focused results dictionary
        focused_results = {
            'dl_depth': depth_dl_normalized,
            'depth_overlay': depth_overlay,
            'stereo_vs_dl_comparison': stereo_vs_dl_comparison,
            'strong_depth_overlay': strong_depth_overlay,
            'anaglyph_rectified': anaglyph_rectified
        }
        
        # Save individual results to separate folders
        print("8. Saving individual results to separate folders...")
        saved_files = self.save_individual_results(focused_results, output_dir)
        
        # Display focused results
        self.display_focused_results(focused_results)
        
        print("✓ Improved processing completed successfully!")
        print("\n📁 Files saved:")
        for file_path in saved_files:
            print(f"   {file_path}")
        
        return focused_results

    def display_focused_results(self, results):
        """Display focused results - only the requested outputs"""
        fig = plt.figure(figsize=(20, 12))
        
        # Row 1: Anaglyph rectified and DL Depth
        plt.subplot(3, 2, 1)
        if results['anaglyph_rectified'] is not None:
            plt.imshow(results['anaglyph_rectified'])
            plt.title('🔴🔵 Improved Anaglyph Stereo Rectification\n(Red=Left, Cyan=Right)', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        plt.subplot(3, 2, 2)
        if results['dl_depth'] is not None:
            plt.imshow(results['dl_depth'], cmap='jet')
            plt.title('🧠 Deep Learning Depth Map', fontsize=12, fontweight='bold')
            plt.colorbar(shrink=0.6)
        plt.axis('off')
        
        # Row 2: DL Depth Overlay and Strong Overlay
        plt.subplot(3, 2, 3)
        if results['depth_overlay'] is not None:
            plt.imshow(results['depth_overlay'])
            plt.title('🎯 DL Depth Overlay (α=0.5)', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        plt.subplot(3, 2, 4)
        if results['strong_depth_overlay'] is not None:
            plt.imshow(results['strong_depth_overlay'])
            plt.title('💪 Strong Depth Overlay (α=0.8)', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # Row 3: Stereo vs DL Comparison (span both columns)
        plt.subplot(3, 1, 3)
        if results['stereo_vs_dl_comparison'] is not None:
            plt.imshow(results['stereo_vs_dl_comparison'])
            plt.title('⚖️ Stereo vs DL Depth Comparison (JET Colormap)', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        # plt.show()
    
    def create_gifs_from_results(self, all_results, output_dir="batch_results", fps=2):
        """Create animated GIFs from all processed results"""
        print("\n" + "="*60)
        print("🎬 CREATING ANIMATED GIFs")
        print("="*60)
        
        if not all_results:
            print("❌ No results available for GIF creation")
            return
        
        # Define which results to create GIFs for
        gif_types = {
            'dl_depth': 'DLDepth_Animation.gif',
            'depth_overlay': 'DLDepthOverlay_Animation.gif', 
            'stereo_vs_dl_comparison': 'StereoVsDL_Animation.gif',
            'strong_depth_overlay': 'StrongDepthOverlay_Animation.gif',
            'anaglyph_rectified': 'AnaglyphRectified_Animation.gif'
        }
        
        created_gifs = []
        
        for result_type, gif_filename in gif_types.items():
            print(f"→ Creating {gif_filename}...")
            
            # Collect all frames for this result type
            frames = []
            valid_frames = 0
            
            for result_data in all_results:
                if result_type in result_data['results'] and result_data['results'][result_type] is not None:
                    image_array = result_data['results'][result_type]
                    
                    # Convert numpy array to PIL Image
                    if len(image_array.shape) == 3:
                        # Color image
                        if image_array.dtype != np.uint8:
                            image_array = (image_array * 255).astype(np.uint8)
                        pil_image = Image.fromarray(image_array)
                    else:
                        # Grayscale image
                        if image_array.dtype != np.uint8:
                            image_array = (image_array * 255).astype(np.uint8)
                        pil_image = Image.fromarray(image_array, mode='L')
                    
                    # Add frame number and pair name as text overlay
                    pil_image_with_text = self.add_text_to_image(
                        pil_image, 
                        f"Frame {result_data['index']}: {result_data['pair_name']}"
                    )
                    
                    frames.append(pil_image_with_text)
                    valid_frames += 1
            
            if valid_frames > 0:
                # Create GIF
                gif_path = Path(output_dir) / gif_filename
                
                # Calculate duration per frame (in milliseconds)
                duration = int(1000 / fps)  # fps to milliseconds
                
                # Save as GIF
                frames[0].save(
                    str(gif_path),
                    save_all=True,
                    append_images=frames[1:],
                    duration=duration,
                    loop=0,  # 0 means infinite loop
                    optimize=True
                )
                
                created_gifs.append(str(gif_path))
                print(f"  ✅ Created: {gif_filename} ({valid_frames} frames, {fps} FPS)")
            else:
                print(f"  ❌ No valid frames for {result_type}")
        
        print(f"\n🎬 Created {len(created_gifs)} animated GIFs:")
        for gif_path in created_gifs:
            print(f"   📁 {gif_path}")
        
        return created_gifs

    def add_text_to_image(self, pil_image, text):
        """Add text overlay to PIL image"""
        # Create a copy to avoid modifying original
        img_with_text = pil_image.copy()
        draw = ImageDraw.Draw(img_with_text)
        
        # Try to use a nice font, fall back to default if not available
        try:
            font_size = max(16, min(pil_image.width // 30, 24))  # Responsive font size
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Get text size for positioning
        if font:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width = len(text) * 8  # Rough estimate
            text_height = 12
        
        # Position text at top-left with padding
        x = 10
        y = 10
        
        # Draw text with black outline for visibility
        outline_width = 2
        for dx in [-outline_width, 0, outline_width]:
            for dy in [-outline_width, 0, outline_width]:
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, fill="black", font=font)
        
        # Draw white text on top
        draw.text((x, y), text, fill="white", font=font)
        
        return img_with_text

    def create_comparison_gif(self, all_results, output_dir="batch_results", fps=1.5):
        """Create a side-by-side comparison GIF showing multiple result types"""
        print("→ Creating comprehensive comparison GIF...")
        
        if not all_results:
            print("❌ No results for comparison GIF")
            return None
        
        comparison_frames = []
        
        for result_data in all_results:
            results = result_data['results']
            frame_title = f"Frame {result_data['index']}: {result_data['pair_name']}"
            
            # Get the main results we want to compare
            dl_depth = results.get('dl_depth')
            depth_overlay = results.get('depth_overlay') 
            anaglyph = results.get('anaglyph_rectified')
            
            if dl_depth is not None and depth_overlay is not None:
                # Create comparison frame
                comparison_frame = self.create_comparison_frame(
                    dl_depth, depth_overlay, anaglyph, frame_title
                )
                
                if comparison_frame is not None:
                    comparison_frames.append(comparison_frame)
        
        if comparison_frames:
            gif_path = Path(output_dir) / "Comprehensive_Comparison.gif"
            duration = int(1000 / fps)
            
            comparison_frames[0].save(
                str(gif_path),
                save_all=True,
                append_images=comparison_frames[1:],
                duration=duration,
                loop=0,
                optimize=True
            )
            
            print(f"  ✅ Created: Comprehensive_Comparison.gif ({len(comparison_frames)} frames)")
            return str(gif_path)
        
        return None

    def create_comparison_frame(self, dl_depth, depth_overlay, anaglyph, title):
        """Create a single comparison frame with multiple views"""
        try:
            # Convert arrays to PIL Images
            if dl_depth is not None:
                dl_pil = Image.fromarray(dl_depth).convert('RGB')
            else:
                dl_pil = None
                
            if depth_overlay is not None:
                overlay_pil = Image.fromarray(depth_overlay)
            else:
                overlay_pil = None
                
            if anaglyph is not None:
                anaglyph_pil = Image.fromarray(anaglyph)
            else:
                anaglyph_pil = None
            
            # Determine frame size based on available images
            available_images = [img for img in [dl_pil, overlay_pil, anaglyph_pil] if img is not None]
            
            if not available_images:
                return None
            
            # Use first available image for size reference
            ref_img = available_images[0]
            img_width, img_height = ref_img.size
            
            # Create comparison layout (horizontal arrangement)
            num_images = len(available_images)
            total_width = img_width * num_images
            total_height = img_height + 40  # Extra space for title
            
            # Create blank canvas
            comparison = Image.new('RGB', (total_width, total_height), (255, 255, 255))
            
            # Add title
            draw = ImageDraw.Draw(comparison)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Center title
            bbox = draw.textbbox((0, 0), title, font=font)
            text_width = bbox[2] - bbox[0]
            title_x = (total_width - text_width) // 2
            draw.text((title_x, 10), title, fill="black", font=font)
            
            # Paste images
            x_offset = 0
            labels = ["DL Depth", "Depth Overlay", "Anaglyph"]
            
            for i, (img, label) in enumerate(zip(available_images, labels)):
                if img is not None:
                    # Resize if necessary
                    if img.size != (img_width, img_height):
                        img = img.resize((img_width, img_height))
                    
                    # Paste image
                    comparison.paste(img, (x_offset, 40))
                    
                    # Add label
                    label_x = x_offset + 10
                    label_y = 40 + img_height - 25
                    
                    # Draw label background
                    label_bbox = draw.textbbox((0, 0), label, font=font)
                    label_bg_width = label_bbox[2] - label_bbox[0] + 8
                    draw.rectangle([label_x-4, label_y-2, label_x+label_bg_width, label_y+18], 
                                fill=(0, 0, 0, 128))
                    
                    # Draw label text
                    draw.text((label_x, label_y), label, fill="white", font=font)
                    
                    x_offset += img_width
            
            return comparison
            
        except Exception as e:
            print(f"Error creating comparison frame: {e}")
            return None
        
    def create_depth_histogram_comparison(self, stereo_depth, dl_depth, pair_name="", save_path=None):
        if stereo_depth is None or dl_depth is None:
            print("❌ Cannot create histogram - one or both depth maps are None")
            return None, None
        
        print(f"→ Creating depth histogram comparison for {pair_name}...")
        
        # Ensure both depth maps have same dimensions
        if stereo_depth.shape != dl_depth.shape:
            stereo_depth = cv2.resize(stereo_depth, (dl_depth.shape[1], dl_depth.shape[0]))
        
        # Remove invalid/zero values for cleaner histograms
        stereo_valid = stereo_depth[stereo_depth > 0].flatten()
        dl_valid = dl_depth[dl_depth > 0].flatten()
        
        # Create histogram comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'📊 Depth Histogram Comparison - {pair_name}', fontsize=16, fontweight='bold')
        
        # 1. Individual histograms
        axes[0, 0].hist(stereo_valid, bins=50, alpha=0.7, color='blue', label='Stereo Depth', density=True)
        axes[0, 0].set_title('🔵 Stereo Depth Distribution')
        axes[0, 0].set_xlabel('Depth Value (0-255)')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(dl_valid, bins=50, alpha=0.7, color='red', label='DL Depth', density=True)
        axes[0, 1].set_title('🔴 DL Depth Distribution') 
        axes[0, 1].set_xlabel('Depth Value (0-255)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 2. Overlapped histograms
        axes[1, 0].hist(stereo_valid, bins=50, alpha=0.6, color='blue', label='Stereo', density=True)
        axes[1, 0].hist(dl_valid, bins=50, alpha=0.6, color='red', label='DL', density=True)
        axes[1, 0].set_title('⚖️ Overlapped Distributions')
        axes[1, 0].set_xlabel('Depth Value (0-255)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 3. Statistical comparison table
        axes[1, 1].axis('off')
        
        # Calculate statistics
        stats = {
            'Stereo': {
                'Valid Pixels': len(stereo_valid),
                'Coverage %': (len(stereo_valid) / stereo_depth.size) * 100,
                'Mean Depth': np.mean(stereo_valid),
                'Std Dev': np.std(stereo_valid),
                'Min': np.min(stereo_valid),
                'Max': np.max(stereo_valid),
                'Median': np.median(stereo_valid)
            },
            'DL': {
                'Valid Pixels': len(dl_valid),
                'Coverage %': (len(dl_valid) / dl_depth.size) * 100,
                'Mean Depth': np.mean(dl_valid),
                'Std Dev': np.std(dl_valid),
                'Min': np.min(dl_valid),
                'Max': np.max(dl_valid),
                'Median': np.median(dl_valid)
            }
        }
        
        # Create statistics table
        table_data = []
        table_data.append(['Metric', 'Stereo', 'DL', 'Difference'])
        table_data.append(['Valid Pixels', f"{stats['Stereo']['Valid Pixels']:,}", 
                        f"{stats['DL']['Valid Pixels']:,}", 
                        f"{stats['DL']['Valid Pixels'] - stats['Stereo']['Valid Pixels']:,}"])
        table_data.append(['Coverage %', f"{stats['Stereo']['Coverage %']:.1f}%", 
                        f"{stats['DL']['Coverage %']:.1f}%",
                        f"{stats['DL']['Coverage %'] - stats['Stereo']['Coverage %']:+.1f}%"])
        table_data.append(['Mean Depth', f"{stats['Stereo']['Mean Depth']:.1f}", 
                        f"{stats['DL']['Mean Depth']:.1f}",
                        f"{stats['DL']['Mean Depth'] - stats['Stereo']['Mean Depth']:+.1f}"])
        table_data.append(['Std Dev', f"{stats['Stereo']['Std Dev']:.1f}", 
                        f"{stats['DL']['Std Dev']:.1f}",
                        f"{stats['DL']['Std Dev'] - stats['Stereo']['Std Dev']:+.1f}"])
        table_data.append(['Range', f"{stats['Stereo']['Max'] - stats['Stereo']['Min']:.1f}", 
                        f"{stats['DL']['Max'] - stats['DL']['Min']:.1f}",
                        f"{(stats['DL']['Max'] - stats['DL']['Min']) - (stats['Stereo']['Max'] - stats['Stereo']['Min']):+.1f}"])
        
        # Display table
        table = axes[1, 1].table(cellText=table_data[1:], colLabels=table_data[0],
                            cellLoc='center', loc='center',
                            colWidths=[0.25, 0.2, 0.2, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color code the header
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        axes[1, 1].set_title('📋 Statistical Comparison', fontweight='bold')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Histogram saved: {save_path}")
        
        return fig, stats

    def analyze_depth_quality_metrics(self, stereo_depth, dl_depth):
        """
        Analyze depth quality metrics for comparison
        
        Returns:
            Dictionary with quality metrics
        """
        metrics = {}
        
        # Remove invalid values
        stereo_valid = stereo_depth[stereo_depth > 0]
        dl_valid = dl_depth[dl_depth > 0]
        
        # Coverage metrics
        metrics['stereo_coverage'] = (len(stereo_valid) / stereo_depth.size) * 100
        metrics['dl_coverage'] = (len(dl_valid) / dl_depth.size) * 100
        metrics['coverage_advantage'] = metrics['dl_coverage'] - metrics['stereo_coverage']
        
        # Distribution metrics
        metrics['stereo_range'] = np.max(stereo_valid) - np.min(stereo_valid) if len(stereo_valid) > 0 else 0
        metrics['dl_range'] = np.max(dl_valid) - np.min(dl_valid) if len(dl_valid) > 0 else 0
        
        # Smoothness (inverse of standard deviation)
        metrics['stereo_smoothness'] = 1 / (np.std(stereo_valid) + 1) if len(stereo_valid) > 0 else 0
        metrics['dl_smoothness'] = 1 / (np.std(dl_valid) + 1) if len(dl_valid) > 0 else 0
        
        return metrics
    
    def generate_quality_summary_report(self, all_results, output_dir):
        """Generate a summary report of depth quality across all processed pairs"""
        
        summary_data = []
        
        for result_data in all_results:
            pair_name = result_data['pair_name']
            results = result_data['results']
            
            # Get depth maps
            stereo_depth = results.get('stereo_depth_normalized')  # You may need to store this in results
            dl_depth = results.get('dl_depth')
            
            if stereo_depth is not None and dl_depth is not None:
                metrics = self.analyze_depth_quality_metrics(stereo_depth, dl_depth)
                metrics['pair_name'] = pair_name
                summary_data.append(metrics)
        
        if summary_data:
            # Create summary plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('📊 Batch Depth Quality Summary', fontsize=16, fontweight='bold')
            
            pair_names = [d['pair_name'] for d in summary_data]
            stereo_coverage = [d['stereo_coverage'] for d in summary_data]
            dl_coverage = [d['dl_coverage'] for d in summary_data]
            
            # Coverage comparison
            x = np.arange(len(pair_names))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, stereo_coverage, width, label='Stereo', alpha=0.7, color='blue')
            axes[0, 0].bar(x + width/2, dl_coverage, width, label='DL', alpha=0.7, color='red')
            axes[0, 0].set_xlabel('Image Pairs')
            axes[0, 0].set_ylabel('Coverage %')
            axes[0, 0].set_title('Depth Coverage Comparison')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(pair_names, rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add more summary plots as needed...
            
            # Save summary report
            summary_path = Path(output_dir) / "DepthQualitySummary.png"
            plt.savefig(str(summary_path), dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ Quality summary saved: {summary_path}")
            
            
# Updated process_multiple_image_pairs method to include GIF creation
    def process_multiple_image_pairs_with_gifs(self, left_folder, right_folder, output_dir="batch_results", create_gifs=True, gif_fps=4):
        """Process multiple image pairs from folders and create GIFs"""
        print("="*80)
        print("BATCH PROCESSING - Multiple Image Pairs + GIF Creation")
        print("="*80)
        
        # Get image pairs
        image_pairs = self.get_image_pairs_from_folders(left_folder, right_folder)
        
        if not image_pairs:
            print("❌ No matching image pairs found!")
            return
        
        all_results = []
        successful_pairs = 0
        
        for i, (left_path, right_path, pair_name) in enumerate(image_pairs, 1):
            print(f"\n{'='*60}")
            print(f"Processing pair {i}/{len(image_pairs)}: {pair_name}")
            print(f"Left: {Path(left_path).name}")
            print(f"Right: {Path(right_path).name}")
            print(f"{'='*60}")
            
            try:
                # Process single image pair
                results = self.process_single_pair(left_path, right_path, output_dir, i, pair_name)
                
                if results:
                    all_results.append({
                        'pair_name': pair_name,
                        'index': i,
                        'results': results,
                        'left_path': left_path,
                        'right_path': right_path
                    })
                    successful_pairs += 1
                    print(f"✅ Successfully processed pair {i}: {pair_name}")
                else:
                    print(f"❌ Failed to process pair {i}: {pair_name}")
                    
            except Exception as e:
                print(f"❌ Error processing pair {i} ({pair_name}): {e}")
                continue
        
        # Create GIFs if requested and we have results
        created_gifs = []
        if create_gifs and successful_pairs > 1:  # Need at least 2 frames for animation
            print(f"\n🎬 Creating animated GIFs from {successful_pairs} processed pairs...")
            created_gifs = self.create_gifs_from_results(all_results, output_dir, gif_fps)
            
            # Also create comprehensive comparison GIF
            comparison_gif = self.create_comparison_gif(all_results, output_dir, gif_fps)
            if comparison_gif:
                created_gifs.append(comparison_gif)
        
        elif create_gifs and successful_pairs <= 1:
            print("⚠️ Need at least 2 processed images to create GIF animations")
        
        if successful_pairs > 0:
            print("→ Generating depth quality summary report...")
            self.generate_quality_summary_report(all_results, output_dir)
            
        # Summary
        print(f"\n{'='*80}")
        print(f"🎯 BATCH PROCESSING COMPLETED!")
        print(f"✅ Successfully processed: {successful_pairs}/{len(image_pairs)} pairs")
        print(f"📁 Static results saved in: {output_dir}/")
        if created_gifs:
            print(f"🎬 Created {len(created_gifs)} animated GIFs:")
            for gif in created_gifs:
                print(f"   📽️ {Path(gif).name}")
        print(f"{'='*80}")
        
        # Show folder structure
        self.show_output_structure_with_gifs(output_dir, successful_pairs, len(created_gifs))
        
        return all_results, created_gifs

    def show_output_structure_with_gifs(self, output_dir, num_pairs, num_gifs):
        """Show the output folder structure including GIFs"""
        print(f"\n📂 Output Structure in '{output_dir}':")
        
        # Static image folders
        folders = ['DLDepth', 'DLDepthOverlay', 'StereoVsDL', 'StrongDepthOverlay', 'AnaglyphRectified', 'HistogramComparisons']
        
        for folder in folders:
            print(f"├── {folder}/")
            for i in range(1, min(num_pairs + 1, 4)):  # Show first 3 examples
                prefix = "│   ├──"
                print(f"{prefix} {folder}{i}.jpg")
            if num_pairs > 3:
                print(f"│   └── ... ({num_pairs} total)")
        
        # GIF files
        if num_gifs > 0:
            print("├── 🎬 Animated GIFs:")
            gif_names = [
                "DLDepth_Animation.gif",
                "DLDepthOverlay_Animation.gif", 
                "StereoVsDL_Animation.gif",
                "StrongDepthOverlay_Animation.gif",
                "AnaglyphRectified_Animation.gif",
                "Comprehensive_Comparison.gif",
                "Histogram_Comparison.gif"
            ]
            
            for gif_name in gif_names:
                print(f"│   ├── 📽️ {gif_name}")
        
        print(f"\n🌈 All depth comparisons use JET colormap")
        print(f"🔴🔵 Anaglyph images show improved alignment")
        if num_gifs > 0:
            print(f"🎬 GIF animations show progression through all {num_pairs} image pairs")

# Usage example and main execution
def main_with_gifs():
    """Main function with GIF creation options"""
    
    # Initialize the processor
    processor = ModifiedRGBDProcessor()
    
    # Folder paths - UPDATE THESE TO YOUR ACTUAL FOLDER PATHS
    left_folder = "frameL"    # Folder containing left images
    right_folder = "frameR"  # Folder containing right images
    output_folder = "RGBD_with_gifs"
    
    # GIF settings
    create_gifs = True  # Set to False if you don't want GIFs
    gif_fps = 6  # Frames per second for GIF animation (1-5 recommended)
    
    # Check if folders exist
    if not os.path.exists(left_folder) or not os.path.exists(right_folder):
        print("❌ Image folders not found. Please update the folder paths.")
        print("Current paths:")
        print(f"  left_folder = '{left_folder}'")
        print(f"  right_folder = '{right_folder}'")
        return
    
    # Process all image pairs and create GIFs
    print("Starting batch processing with GIF creation...")
    results, gifs = processor.process_multiple_image_pairs_with_gifs(
        left_folder, 
        right_folder, 
        output_dir=output_folder,
        create_gifs=create_gifs,
        gif_fps=gif_fps
    )
    
    if results:
        print("\n" + "="*80)
        print("🎯 BATCH PROCESSING WITH GIFS COMPLETED!")
        print(f"📁 Static images saved in subfolders of: {output_folder}/")
        if gifs:
            print(f"🎬 {len(gifs)} animated GIFs created in: {output_folder}/")
            print("   📽️ Each GIF shows the progression through all your image pairs")
            print(f"   ⏱️ Animation speed: {gif_fps} FPS")
        print("="*80)
    else:
        print("❌ No images were processed successfully!")
        
if __name__ == "__main__":
    main_with_gifs()