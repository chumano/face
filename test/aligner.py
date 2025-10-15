"""
Reconstructed Python code from face_aligner.so disassembly
Module: processing.face_aligner
This is a face alignment and preprocessing module using OpenCV and scikit-image
Generated from assembly analysis
"""

import cv2
import numpy as np
from skimage.transform import SimilarityTransform
import math


class FaceAligner:
    """
    Face aligner class for preprocessing face images
    Uses similarity transformation for face alignment based on landmarks
    """
    
    def __init__(self, image_size=112, margin=0):
        """
        Initialize the FaceAligner
        
        Args:
            image_size: Target size for the aligned face image (default: 112)
            margin: Margin to add around the face bounding box (default: 0)
        """
        self.image_size = image_size
        self.margin = margin
        
        # Standard facial landmarks reference points for alignment
        # These are typical 5-point landmarks (2 eyes, nose, 2 mouth corners)
        # Normalized to image_size
        self.reference_landmarks = None
        self._init_reference_landmarks()
    
    def _init_reference_landmarks(self):
        """
        Initialize reference facial landmarks for alignment
        Standard 5-point face model
        """
        # Standard face template landmarks (normalized coordinates)
        # Format: [[left_eye], [right_eye], [nose], [left_mouth], [right_mouth]]
        image_size = self.image_size
        
        # Typical landmark positions for a frontal face
        # These are approximate normalized positions
        self.reference_landmarks = np.array([
            [30.2946 + 8.0, 51.6963],  # left eye
            [65.5318 + 8.0, 51.5014],  # right eye
            [48.0252 + 8.0, 71.7366],  # nose tip
            [33.5493 + 8.0, 92.3655],  # left mouth corner
            [62.7299 + 8.0, 92.2041]   # right mouth corner
        ], dtype=np.float32)
        
        # Scale to image size if not 112
        if image_size != 112:
            scale = image_size / 112.0
            self.reference_landmarks = self.reference_landmarks * scale
    
    def preprocess(self, img, bbox=None, landmark=None):
        """
        Preprocess and align face image based on landmarks
        
        Args:
            img: Input image (numpy array)
            bbox: Face bounding box [x1, y1, x2, y2] (optional)
            landmark: Facial landmarks, typically 5 points [[x,y], ...] (optional)
            
        Returns:
            Aligned and cropped face image
        """
        if landmark is not None:
            # Perform alignment using landmarks
            return self._align_face_with_landmarks(img, landmark)
        elif bbox is not None:
            # Crop using bounding box
            return self._crop_image(img, bbox)
        else:
            # Return resized image
            return cv2.resize(img, (self.image_size, self.image_size))
    
    def _align_face_with_landmarks(self, img, landmark):
        """
        Align face using facial landmarks via similarity transformation
        
        Args:
            img: Input image
            landmark: Facial landmarks (Nx2 array)
            
        Returns:
            Aligned face image
        """
        # Ensure landmark is numpy array
        landmark = np.array(landmark, dtype=np.float32)
        
        # Estimate similarity transform from landmarks to reference
        tform = SimilarityTransform()
        tform.estimate(landmark, self.reference_landmarks)
        
        # Get transformation matrix
        M = tform.params[0:2, :]
        
        # Apply affine transformation
        warped = cv2.warpAffine(
            img, 
            M, 
            (self.image_size, self.image_size),
            borderValue=0.0
        )
        
        return warped
    
    def _crop_image(self, img, bbox):
        """
        Crop image using bounding box with optional margin
        
        Args:
            img: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Cropped and resized face image
        """
        bbox = np.array(bbox, dtype=np.int32)
        
        # Extract bounding box coordinates
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # Add margin
        if self.margin > 0:
            w = x2 - x1
            h = y2 - y1
            x1 = max(0, x1 - int(w * self.margin))
            y1 = max(0, y1 - int(h * self.margin))
            x2 = min(img.shape[1], x2 + int(w * self.margin))
            y2 = min(img.shape[0], y2 + int(h * self.margin))
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, img.shape[1]))
        y1 = max(0, min(y1, img.shape[0]))
        x2 = max(0, min(x2, img.shape[1]))
        y2 = max(0, min(y2, img.shape[0]))
        
        # Crop image
        cropped = img[y1:y2, x1:x2]
        
        # Resize to target size
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            cropped_image = cv2.resize(cropped, (self.image_size, self.image_size))
        else:
            # Return zeros if crop is invalid
            if len(img.shape) == 3:
                cropped_image = np.zeros((self.image_size, self.image_size, img.shape[2]), dtype=img.dtype)
            else:
                cropped_image = np.zeros((self.image_size, self.image_size), dtype=img.dtype)
        
        return cropped_image
    
    def crop_image(self, img, bbox):
        """
        Public method to crop image (alias for _crop_image)
        
        Args:
            img: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Cropped and resized face image
        """
        return self._crop_image(img, bbox)


def test():
    """
    Simple test function for the FaceAligner class
    """
    # Create aligner
    aligner = FaceAligner(image_size=112, margin=0.1)
    
    # Create dummy image
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test with bounding box
    bbox = [100, 100, 300, 300]
    result = aligner.preprocess(img, bbox=bbox)
    print(f"Preprocessed with bbox: {result.shape}")
    
    # Test with landmarks
    landmarks = np.array([
        [150, 180],  # left eye
        [250, 180],  # right eye
        [200, 220],  # nose
        [170, 270],  # left mouth
        [230, 270]   # right mouth
    ], dtype=np.float32)
    
    result = aligner.preprocess(img, landmark=landmarks)
    print(f"Preprocessed with landmarks: {result.shape}")
    
    return aligner, result


if __name__ == "__main__":
    aligner, result = test()
    print("FaceAligner test completed successfully!")
