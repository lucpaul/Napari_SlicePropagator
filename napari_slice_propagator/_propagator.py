"""
Core functionality for slice propagation and active contour segmentation.
"""

import numpy as np
from skimage import segmentation, measure, morphology, filters
from skimage.segmentation import active_contour, morphological_chan_vese, find_boundaries
from scipy import ndimage
from typing import Tuple, Optional, Dict, Any, Set, List
import warnings

class SlicePropagator:
    """Handles slice-to-slice annotation propagation and active contour segmentation."""
    
    def __init__(self):
        # Updated refinement methods
        self.refinement_methods = {
            'snake': self._apply_snake,
            'chan_vese': self._apply_chan_vese,
            'intensity_snap': self._apply_intensity_snap,
            'graph_cut': self._apply_graph_cut
        }
        
        # Metadata tracking for manual vs auto labels
        # Structure: {slice_idx: {'manual': set([label_ids]), 'auto': set([label_ids]), 'all_existing': set([label_ids])}}
        self.label_metadata = {}
        
        # Snapshots for change detection
        # Structure: {slice_idx: np.ndarray}
        self.slice_snapshots = {}
        
        # Track object mappings across slices for label merging
        # Structure: {object_id: {slice_idx: label_id, ...}}
        self.object_mappings = {}
        self.next_object_id = 1
    
    def initialize_metadata_for_slice(self, slice_idx: int, labels_slice: np.ndarray):
        """Initialize metadata tracking for a slice."""
        if slice_idx not in self.label_metadata:
            self.label_metadata[slice_idx] = {'manual': set(), 'auto': set(), 'all_existing': set()}
        
        # Get all existing labels
        unique_labels = set(np.unique(labels_slice))
        unique_labels.discard(0)  # Remove background
        
        # If slice has existing labels, consider them manual (unless we know they're auto)
        if unique_labels:
            # Only mark as manual if they're not already tracked as auto
            new_manual_labels = unique_labels - self.label_metadata[slice_idx]['auto']
            self.label_metadata[slice_idx]['manual'].update(new_manual_labels)
            self.label_metadata[slice_idx]['all_existing'].update(unique_labels)
        
        # Store snapshot
        self.slice_snapshots[slice_idx] = labels_slice.copy()
    
    def detect_manual_changes(self, slice_idx: int, current_labels: np.ndarray):
        """
        Detect manual changes by comparing current state with snapshot.
        Updates metadata to mark changed labels as manual.
        """
        if slice_idx not in self.slice_snapshots:
            self.initialize_metadata_for_slice(slice_idx, current_labels)
            return
        
        previous_labels = self.slice_snapshots[slice_idx]
        
        # Find labels that have changed
        if not np.array_equal(current_labels, previous_labels):
            # Get current unique labels
            current_unique = set(np.unique(current_labels))
            current_unique.discard(0)
            
            previous_unique = set(np.unique(previous_labels))
            previous_unique.discard(0)
            
            # New labels are manual
            new_labels = current_unique - previous_unique
            if new_labels:
                self.label_metadata[slice_idx]['manual'].update(new_labels)
                self.label_metadata[slice_idx]['auto'].difference_update(new_labels)
                self.label_metadata[slice_idx]['all_existing'].update(new_labels)
            
            # Check for modified existing labels
            for label_id in current_unique.intersection(previous_unique):
                current_mask = (current_labels == label_id)
                previous_mask = (previous_labels == label_id)
                
                # If this label has changed, mark as manual
                if not np.array_equal(current_mask, previous_mask):
                    self.label_metadata[slice_idx]['manual'].add(label_id)
                    self.label_metadata[slice_idx]['auto'].discard(label_id)
            
            # Update all_existing
            self.label_metadata[slice_idx]['all_existing'] = current_unique.copy()
            
            # Update snapshot
            self.slice_snapshots[slice_idx] = current_labels.copy()
    
    def mark_labels_as_auto(self, slice_idx: int, label_ids: Set[int]):
        """Mark specific labels as auto-generated."""
        if slice_idx not in self.label_metadata:
            self.label_metadata[slice_idx] = {'manual': set(), 'auto': set(), 'all_existing': set()}
        
        self.label_metadata[slice_idx]['auto'].update(label_ids)
        self.label_metadata[slice_idx]['manual'].difference_update(label_ids)
        self.label_metadata[slice_idx]['all_existing'].update(label_ids)
    
    def get_auto_labels_for_slice(self, slice_idx: int) -> Set[int]:
        """Get set of auto-generated label IDs for a slice."""
        if slice_idx not in self.label_metadata:
            return set()
        return self.label_metadata[slice_idx]['auto'].copy()
    
    def get_manual_labels_for_slice(self, slice_idx: int) -> Set[int]:
        """Get set of manually created label IDs for a slice."""
        if slice_idx not in self.label_metadata:
            return set()
        return self.label_metadata[slice_idx]['manual'].copy()
    
    def get_all_existing_labels_for_slice(self, slice_idx: int) -> Set[int]:
        """Get set of all existing label IDs for a slice."""
        if slice_idx not in self.label_metadata:
            return set()
        return self.label_metadata[slice_idx]['all_existing'].copy()
    
    def has_annotations(self, labels_slice: np.ndarray) -> bool:
        """Check if a label slice contains any annotations."""
        return np.any(labels_slice > 0)

    def _filter_small_labels(self, labels_slice: np.ndarray, min_pixels: int = 15) -> np.ndarray:
        """
        Remove labels with fewer than min_pixels pixels.
        
        Parameters:
        -----------
        labels_slice : np.ndarray
            2D labeled image
        min_pixels : int
            Minimum number of pixels required for a label to be kept
            
        Returns:
        --------
        np.ndarray
            Filtered labeled image
        """
        if not self.has_annotations(labels_slice):
            return labels_slice
        
        result = labels_slice.copy()
        unique_labels = np.unique(labels_slice)
        unique_labels = unique_labels[unique_labels > 0]
        
        for label_id in unique_labels:
            label_mask = (labels_slice == label_id)
            pixel_count = np.sum(label_mask)
            
            if pixel_count < min_pixels:
                result[label_mask] = 0  # Remove small label
        
        return result
    

    def get_new_labels_to_propagate(self, from_slice_idx: int, to_slice_idx: int, 
                                labels_layer_data: np.ndarray, overlap_threshold: float = 0.1) -> Set[int]:
        """
        Get labels from source slice that are not already present on target slice.
        Only propagate labels that don't exist on target slice at all.
        """
        source_labels = labels_layer_data[from_slice_idx]
        target_labels = labels_layer_data[to_slice_idx]
        
        # Get all labels from source
        source_unique = set(np.unique(source_labels))
        source_unique.discard(0)
        
        # Get existing labels on target - ANY existing label prevents propagation
        target_unique = set(np.unique(target_labels))
        target_unique.discard(0)
        
        if not source_unique:
            return set()
        
        # Only propagate labels that have NO overlap with any existing target regions
        labels_to_propagate = set()
        
        for source_label in source_unique:
            source_mask = (source_labels == source_label)
            
            # Check if ANY pixels of this source region overlap with ANY existing target region
            overlap_check = target_labels[source_mask]
            has_any_overlap = np.any(overlap_check > 0)
            
            # Only propagate if there's NO overlap at all
            if not has_any_overlap:
                labels_to_propagate.add(source_label)
        
        return labels_to_propagate


    # def get_new_labels_to_propagate(self, from_slice_idx: int, to_slice_idx: int, 
    #                                labels_layer_data: np.ndarray, overlap_threshold: float = 0.1) -> Set[int]:
    #     """
    #     Get labels from source slice that are not already present on target slice.
    #     Lower overlap threshold to allow propagation to slices with existing labels.
    #     """
    #     source_labels = labels_layer_data[from_slice_idx]
    #     target_labels = labels_layer_data[to_slice_idx]
        
    #     # Get all labels from source
    #     source_unique = set(np.unique(source_labels))
    #     source_unique.discard(0)
        
    #     # Get existing labels on target
    #     target_unique = set(np.unique(target_labels))
    #     target_unique.discard(0)
        
    #     if not source_unique:
    #         return set()
        
    #     # For incremental propagation, check which source regions
    #     # don't have significant overlap with existing target regions
    #     labels_to_propagate = set()
        
    #     for source_label in source_unique:
    #         source_mask = (source_labels == source_label)
            
    #         # Check if this region overlaps significantly with any existing target region
    #         has_significant_overlap = False
    #         if target_unique:
    #             overlap_check = target_labels[source_mask]
    #             unique_overlaps = set(np.unique(overlap_check))
    #             unique_overlaps.discard(0)
                
    #             # Calculate overlap ratio - use lower threshold for better propagation
    #             if unique_overlaps:
    #                 total_source_pixels = np.sum(source_mask)
    #                 overlapping_pixels = np.sum(overlap_check > 0)
    #                 overlap_ratio = overlapping_pixels / total_source_pixels if total_source_pixels > 0 else 0
                    
    #                 # Use lower overlap threshold to allow more propagation
    #                 if overlap_ratio > overlap_threshold:
    #                     has_significant_overlap = True
            
    #         if not has_significant_overlap:
    #             labels_to_propagate.add(source_label)
        
    #     return labels_to_propagate
    
    def incremental_propagate(self, from_slice_idx: int, to_slice_idx: int, 
                            labels_layer_data: np.ndarray, apply_postprocessing: bool = True) -> np.ndarray:
        """
        Incrementally propagate only new labels from source to target slice.
        This allows adding new labels to slices that already have annotations.
        """
        # Initialize metadata if needed
        source_labels = labels_layer_data[from_slice_idx]
        target_labels = labels_layer_data[to_slice_idx]
        
        self.initialize_metadata_for_slice(from_slice_idx, source_labels)
        self.initialize_metadata_for_slice(to_slice_idx, target_labels)
        
        # Get labels that need to be propagated (with lower overlap threshold)
        labels_to_propagate = self.get_new_labels_to_propagate(
            from_slice_idx, to_slice_idx, labels_layer_data, overlap_threshold=0.1
        )
        
        if not labels_to_propagate:
            return labels_layer_data
        
        # Start with a copy of the data
        updated_data = labels_layer_data.copy()
        
        # Find the maximum label ID across all slices to avoid conflicts
        max_label_id = 0
        for slice_idx in range(labels_layer_data.shape[0]):
            slice_labels = set(np.unique(labels_layer_data[slice_idx]))
            slice_labels.discard(0)
            if slice_labels:
                max_label_id = max(max_label_id, max(slice_labels))
        
        # Create mapping from source labels to new label IDs
        label_mapping = {}
        new_auto_labels = set()
        next_label_id = max_label_id + 1
        
        # Propagate only the new labels
        for old_label in labels_to_propagate:
            label_mapping[old_label] = next_label_id
            new_auto_labels.add(next_label_id)
            next_label_id += 1
        
        # Apply the mapping to propagate new labels
        target_slice = updated_data[to_slice_idx].copy()
        for old_label, new_label in label_mapping.items():
            source_mask = (source_labels == old_label)
            target_slice[source_mask] = new_label
        
        # Update the data
        updated_data[to_slice_idx] = target_slice

        # Only apply postprocessing if requested (skip when refinement will be applied)
        if apply_postprocessing:
            # Filter out small labels (less than 15 pixels)
            target_slice = self._filter_small_labels(target_slice, min_pixels=15)

            # Apply automatic merging with minimal threshold (boundary touching)
            target_slice = self.merge_overlapping_labels_within_slice(target_slice, overlap_threshold=0.05)

            # Update the data with filtered and merged labels
            updated_data[to_slice_idx] = target_slice
        
        # Mark new labels as auto-generated
        self.mark_labels_as_auto(to_slice_idx, new_auto_labels)
        
        # Update snapshot
        self.slice_snapshots[to_slice_idx] = target_slice.copy()
        
        return updated_data
    
    def batch_propagate(self, from_slice_idx: int, labels_layer_data: np.ndarray, 
                       n_slices: int, direction: str = 'forward', 
                       apply_refinement: bool = False, refinement_method: str = 'snake',
                       refinement_params: Optional[Dict[str, Any]] = None,
                       image_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Batch propagate labels from a slice to multiple subsequent slices.
        
        Parameters:
        -----------
        from_slice_idx : int
            Source slice index
        labels_layer_data : np.ndarray
            3D labels data
        n_slices : int
            Number of slices to propagate to
        direction : str
            'forward' or 'backward'
        apply_refinement : bool
            Whether to apply refinement after propagation
        refinement_method : str
            Method to use for refinement
        refinement_params : dict
            Parameters for refinement method
        image_data : np.ndarray
            3D image data (required if apply_refinement is True)
        
        Returns:
        --------
        np.ndarray
            Updated labels data
        """
        updated_data = labels_layer_data.copy()
        
        # Determine slice range
        if direction == 'forward':
            target_slices = range(from_slice_idx + 1, 
                                min(from_slice_idx + 1 + n_slices, labels_layer_data.shape[0]))
        else:  # backward
            target_slices = range(from_slice_idx - 1, 
                                max(from_slice_idx - 1 - n_slices, -1), -1)
        
        for target_slice in target_slices:
            # Determine the source slice for this iteration
            if target_slice == list(target_slices)[0]:
            # First iteration: use the original source slice
                source_slice = from_slice_idx
            else:
                # Subsequent iterations: use the previous slice as source
                source_slice = target_slice - 1 if direction == 'forward' else target_slice + 1
    
            # Propagate from source slice to this target slice
            updated_data = self.incremental_propagate(source_slice, target_slice, updated_data, 
                                                    apply_postprocessing=not apply_refinement)

            # Apply refinement if requested
            if apply_refinement and image_data is not None:
                target_labels = updated_data[target_slice]
                if self.has_annotations(target_labels):
                    refined_labels = self.apply_refinement_method(
                        image_data[target_slice], target_labels, 
                        refinement_method, refinement_params
                    )
                    
                    # Apply postprocessing AFTER refinement
                    refined_labels = self.apply_postprocessing(refined_labels)
                    updated_data[target_slice] = refined_labels

                    # Update snapshot
                    self.slice_snapshots[target_slice] = refined_labels.copy()
            # # Propagate from source slice to this target slice
            # updated_data = self.incremental_propagate(source_slice, target_slice, updated_data)
    
            # # Apply refinement if requested
            # if apply_refinement and image_data is not None:
            #     target_labels = updated_data[target_slice]
            #     if self.has_annotations(target_labels):
            #         refined_labels = self.apply_refinement_method(
            #             image_data[target_slice], target_labels, 
            #             refinement_method, refinement_params
            #         )
            #         updated_data[target_slice] = refined_labels
            
            #         # Update snapshot
            #         self.slice_snapshots[target_slice] = refined_labels.copy()
        
        return updated_data
    

    def apply_postprocessing(self, labels_slice: np.ndarray, min_pixels: int = 15, 
                        merge_threshold: float = 0.05) -> np.ndarray:
        """
        Apply postprocessing: filter small labels and merge touching labels.
        
        Parameters:
        -----------
        labels_slice : np.ndarray
            2D labeled image
        min_pixels : int
            Minimum number of pixels required for a label to be kept
        merge_threshold : float
            Threshold for merging touching labels
            
        Returns:
        --------
        np.ndarray
            Postprocessed labeled image
        """
        if not self.has_annotations(labels_slice):
            return labels_slice
        
        # Filter out small labels
        result = self._filter_small_labels(labels_slice, min_pixels=min_pixels)
        
        # Apply automatic merging
        result = self.merge_overlapping_labels_within_slice(result, overlap_threshold=merge_threshold)
        
        return result

    def apply_morphological_operation(self, labels: np.ndarray, operation: str, 
                                    kernel_size: int = 3) -> np.ndarray:
        """
        Apply morphological operations to all labels in a slice.
        
        Parameters:
        -----------
        labels : np.ndarray
            2D labeled image
        operation : str
            'dilation', 'erosion', 'opening', 'closing'
        kernel_size : int
            Size of the morphological kernel
            
        Returns:
        --------
        np.ndarray
            Modified labeled image
        """
        if not self.has_annotations(labels):
            return labels
        
        kernel = morphology.disk(kernel_size)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]
        
        result = np.zeros_like(labels)
        
        for label_id in unique_labels:
            # Create binary mask for this label
            binary_mask = (labels == label_id)
            
            # Apply morphological operation
            if operation == 'dilation':
                modified_mask = morphology.binary_dilation(binary_mask, kernel)
            elif operation == 'erosion':
                modified_mask = morphology.binary_erosion(binary_mask, kernel)
            elif operation == 'opening':
                modified_mask = morphology.binary_opening(binary_mask, kernel)
            elif operation == 'closing':
                modified_mask = morphology.binary_closing(binary_mask, kernel)
            else:
                modified_mask = binary_mask
            
            # Add to result (handle overlaps by keeping original label)
            result[modified_mask & (result == 0)] = label_id
        
        return result

    def merge_overlapping_labels_within_slice(self, labels_slice: np.ndarray,
                                            overlap_threshold: float = 0.05) -> np.ndarray:
        """
        Merge adjacent labels within a single slice based on boundary contact.
        
        Parameters:
        -----------
        labels_slice : np.ndarray
            2D labeled image
        overlap_threshold : float
            Minimum boundary contact ratio to trigger merging (0-1)
            
        Returns:
        --------
        np.ndarray
            Modified slice with merged labels
        """
        # Check if there are any labels
        if labels_slice.max() == 0:
            return labels_slice
            
        result = labels_slice.copy()
        unique_labels = np.unique(labels_slice)
        unique_labels = unique_labels[unique_labels > 0]
        
        if len(unique_labels) < 2:
            return result
        
        # Track which labels have been merged
        merged_labels = set()
        
        for i, label1 in enumerate(unique_labels):
            if label1 in merged_labels:
                continue
                
            mask1 = (result == label1)  # Use result, not original labels_slice
            
            # Get the boundary of label1
            boundary1 = find_boundaries(mask1, mode='inner')
            
            for j, label2 in enumerate(unique_labels[i+1:], i+1):
                if label2 in merged_labels:
                    continue
                    
                mask2 = (result == label2)  # Use result, not original labels_slice
                
                # Get the boundary of label2
                boundary2 = find_boundaries(mask2, mode='inner')
                
                # Find adjacent pixels (boundary pixels that are neighbors)
                # Dilate boundaries by 1 pixel to find adjacency
                dilated_boundary1 = ndimage.binary_dilation(boundary1)
                dilated_boundary2 = ndimage.binary_dilation(boundary2)
                
                # Find contact: where dilated boundary1 intersects with boundary2
                contact = dilated_boundary1 & boundary2
                contact_length = np.sum(contact)
                
                if contact_length > 0:
                    # Calculate contact ratio relative to the smaller boundary
                    boundary1_length = np.sum(boundary1)
                    boundary2_length = np.sum(boundary2)
                    smaller_boundary = min(boundary1_length, boundary2_length)
                    
                    contact_ratio = contact_length / smaller_boundary if smaller_boundary > 0 else 0
                    
                    if contact_ratio >= overlap_threshold:
                        # Merge label2 into label1
                        result[mask2] = label1
                        merged_labels.add(label2)
        
        return result



    # def merge_overlapping_labels_within_slice(self, labels_slice: np.ndarray, 
    #                                         overlap_threshold: float = 0.5) -> np.ndarray:
    #     """
    #     Merge overlapping labels within a single slice.
        
    #     Parameters:
    #     -----------
    #     labels_slice : np.ndarray
    #         2D labeled image
    #     overlap_threshold : float
    #         Minimum overlap ratio to trigger merging
            
    #     Returns:
    #     --------
    #     np.ndarray
    #         Modified slice with merged labels
    #     """
    #     if not self.has_annotations(labels_slice):
    #         return labels_slice
        
    #     result = labels_slice.copy()
    #     unique_labels = np.unique(labels_slice)
    #     unique_labels = unique_labels[unique_labels > 0]
        
    #     if len(unique_labels) < 2:
    #         return result
        
    #     # Track which labels have been merged
    #     merged_labels = set()
        
    #     for i, label1 in enumerate(unique_labels):
    #         if label1 in merged_labels:
    #             continue
                
    #         mask1 = (labels_slice == label1)
    #         label1_size = np.sum(mask1)
            
    #         for j, label2 in enumerate(unique_labels[i+1:], i+1):
    #             if label2 in merged_labels:
    #                 continue
                    
    #             mask2 = (labels_slice == label2)
    #             label2_size = np.sum(mask2)
                
    #             # Calculate intersection
    #             intersection = np.sum(mask1 & mask2)
                
    #             if intersection > 0:
    #                 # Calculate overlap as intersection / smaller region
    #                 smaller_size = min(label1_size, label2_size)
    #                 overlap_ratio = intersection / smaller_size
                    
    #                 if overlap_ratio >= overlap_threshold:
    #                     # Merge label2 into label1
    #                     result[mask2] = label1
    #                     merged_labels.add(label2)
    #                     # Update mask1 to include merged region for subsequent comparisons
    #                     mask1 = (result == label1)
    #                     label1_size = np.sum(mask1)
        
    #     return result
    
    def merge_labels_across_slices(self, labels_layer_data: np.ndarray, 
                                  similarity_threshold: float = 0.7) -> np.ndarray:
        """
        Merge labels across slices so that the same object has the same ID throughout the stack.
        """
        # Store original manual/auto status before merging
        original_manual_status = {}
        original_auto_status = {}
        merged_data = labels_layer_data.copy()
        n_slices = merged_data.shape[0]
        
        for slice_idx in range(n_slices):
            if slice_idx in self.label_metadata:
                original_manual_status[slice_idx] = self.label_metadata[slice_idx]['manual'].copy()
                original_auto_status[slice_idx] = self.label_metadata[slice_idx]['auto'].copy()
        

        
        # Track object mappings: {object_id: [(slice_idx, original_label_id), ...]}
        object_mappings = {}
        next_object_id = 1
        
        # Process each slice
        for slice_idx in range(n_slices):
            current_slice = merged_data[slice_idx]
            unique_labels = set(np.unique(current_slice))
            unique_labels.discard(0)
            
            if not unique_labels:
                continue
            
            # For the first slice, create new object IDs
            if slice_idx == 0:
                label_to_object = {}
                for label_id in unique_labels:
                    object_mappings[next_object_id] = [(slice_idx, label_id)]
                    label_to_object[label_id] = next_object_id
                    next_object_id += 1
                
                # Apply new object IDs
                new_slice = current_slice.copy()
                for old_label, object_id in label_to_object.items():
                    new_slice[current_slice == old_label] = object_id
                merged_data[slice_idx] = new_slice
                
            else:
                # For subsequent slices, match with previous slice objects
                prev_slice = merged_data[slice_idx - 1]
                label_to_object = {}
                
                for label_id in unique_labels:
                    current_mask = (current_slice == label_id)
                    
                    # Find best matching object from previous slice
                    best_match_object_id = None
                    best_overlap = 0
                    
                    prev_unique = set(np.unique(prev_slice))
                    prev_unique.discard(0)
                    
                    for prev_object_id in prev_unique:
                        prev_mask = (prev_slice == prev_object_id)
                        
                        # Calculate overlap
                        intersection = np.sum(current_mask & prev_mask)
                        union = np.sum(current_mask | prev_mask)
                        
                        if union > 0:
                            overlap = intersection / union
                            
                            if overlap > best_overlap and overlap >= similarity_threshold:
                                best_overlap = overlap
                                best_match_object_id = prev_object_id
                    
                    if best_match_object_id is not None:
                        # Match found - use existing object ID
                        label_to_object[label_id] = best_match_object_id
                        object_mappings[best_match_object_id].append((slice_idx, label_id))
                    else:
                        # No match found - create new object ID
                        object_mappings[next_object_id] = [(slice_idx, label_id)]
                        label_to_object[label_id] = next_object_id
                        next_object_id += 1
                
                # Apply object IDs
                new_slice = current_slice.copy()
                for old_label, object_id in label_to_object.items():
                    new_slice[current_slice == old_label] = object_id
                merged_data[slice_idx] = new_slice

            # Restore manual/auto status for merged labels
        for slice_idx in range(n_slices):
            if slice_idx not in self.label_metadata:
                continue
        
            # Clear current status
            self.label_metadata[slice_idx]['manual'].clear()
            self.label_metadata[slice_idx]['auto'].clear()
    
            current_slice = merged_data[slice_idx]
            current_unique = set(np.unique(current_slice))
            current_unique.discard(0)
            
            # For each current label, check if it came from a manual or auto label
            for current_label in current_unique:
                current_mask = (current_slice == current_label)
                
                # Check which original labels this current label represents
                if slice_idx in original_manual_status:
                    for orig_label in original_manual_status[slice_idx]:
                        # Check if this current label overlaps significantly with original manual label
                        if slice_idx < labels_layer_data.shape[0]:
                            orig_slice = labels_layer_data[slice_idx]
                            orig_mask = (orig_slice == orig_label)
                            overlap = np.sum(current_mask & orig_mask)
                            if overlap > 0:
                                self.label_metadata[slice_idx]['manual'].add(current_label)
                                break
                
                # If not marked as manual, check if it was auto
                if (current_label not in self.label_metadata[slice_idx]['manual'] and 
                    slice_idx in original_auto_status):
                    for orig_label in original_auto_status[slice_idx]:
                        if slice_idx < labels_layer_data.shape[0]:
                            orig_slice = labels_layer_data[slice_idx]
                            orig_mask = (orig_slice == orig_label)
                            overlap = np.sum(current_mask & orig_mask)
                            if overlap > 0:
                                self.label_metadata[slice_idx]['auto'].add(current_label)
                                break
            
            # Update all_existing
            self.label_metadata[slice_idx]['all_existing'] = current_unique.copy()
        
        return merged_data
    
    def labels_to_contours(self, labels_slice: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """
        Convert labeled regions to contour coordinates, preserving label IDs.
        
        Parameters:
        -----------
        labels_slice : np.ndarray
            2D labeled image
            
        Returns:
        --------
        list
            List of (label_id, contour) tuples
        """
        contours = []
        
        # Find unique labels (excluding background)
        unique_labels = np.unique(labels_slice)
        unique_labels = unique_labels[unique_labels > 0]
        
        for label_id in unique_labels:
            # Create binary mask for this label
            binary_mask = (labels_slice == label_id)
            
            # Find contours
            try:
                contour_coords = measure.find_contours(binary_mask.astype(float), 0.5)
                if contour_coords:
                    # Take the longest contour (main object boundary)
                    main_contour = max(contour_coords, key=len)
                    if len(main_contour) > 4:  # Minimum points for active contour
                        contours.append((label_id, main_contour))
            except Exception as e:
                warnings.warn(f"Could not extract contour for label {label_id}: {e}")
                continue
                
        return contours
    
    def contours_to_labels(self, contours: List[Tuple[int, np.ndarray]], 
                          image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert contours back to labeled image, preserving original label IDs.
        
        Parameters:
        -----------
        contours : list
            List of (label_id, contour) tuples
        image_shape : tuple
            Shape of the output image
            
        Returns:
        --------
        np.ndarray
            2D labeled image
        """
        labels = np.zeros(image_shape, dtype=np.int32)
        
        for label_id, contour in contours:
            if len(contour) < 3:
                continue
                
            # Create binary mask from contour
            mask = np.zeros(image_shape, dtype=bool)
            
            # Convert contour to integer coordinates
            contour_int = np.round(contour).astype(int)
            
            # Ensure coordinates are within bounds
            contour_int[:, 0] = np.clip(contour_int[:, 0], 0, image_shape[0] - 1)
            contour_int[:, 1] = np.clip(contour_int[:, 1], 0, image_shape[1] - 1)
            
            # Fill the contour
            from skimage.draw import polygon
            try:
                rr, cc = polygon(contour_int[:, 0], contour_int[:, 1], image_shape)
                mask[rr, cc] = True
                # Use original label ID instead of sequential numbering
                labels[mask] = label_id
            except Exception as e:
                warnings.warn(f"Could not create mask from contour for label {label_id}: {e}")
                continue
                
        return labels
    
    
    def _apply_intensity_snap(self, image: np.ndarray, initial_contour: np.ndarray, 
                            params: Dict[str, Any]) -> np.ndarray:
        """Snap contour points to nearby intensity edges."""
        try:
            # Calculate gradient magnitude
            gradient = filters.sobel(image)
            
            # Parameters
            search_radius = params.get('search_radius', 5)
            edge_threshold = params.get('edge_threshold', 0.1)
            
            refined_contour = initial_contour.copy()
            
            # For each contour point, find nearby strong edges
            for i, point in enumerate(initial_contour):
                y, x = int(round(point[0])), int(round(point[1]))
                
                # Define search region
                y_min = max(0, y - search_radius)
                y_max = min(image.shape[0], y + search_radius + 1)
                x_min = max(0, x - search_radius)
                x_max = min(image.shape[1], x + search_radius + 1)
                
                # Find strongest edge in search region
                search_region = gradient[y_min:y_max, x_min:x_max]
                
                if search_region.size > 0:
                    max_gradient_idx = np.unravel_index(np.argmax(search_region), search_region.shape)
                    max_gradient_value = search_region[max_gradient_idx]
                    
                    # Only snap if edge is strong enough
                    if max_gradient_value > edge_threshold:
                        new_y = y_min + max_gradient_idx[0]
                        new_x = x_min + max_gradient_idx[1]
                        refined_contour[i] = [new_y, new_x]

            
        except Exception as e:
            warnings.warn(f"Intensity snapping failed: {e}")
            return initial_contour

        if params.get('apply_builtin_erosion', True):
        # Convert contour to mask, erode, then back to contour
            try:
                mask = np.zeros(image.shape, dtype=bool)
                from skimage.draw import polygon
                contour_int = np.round(refined_contour).astype(int)
                contour_int[:, 0] = np.clip(contour_int[:, 0], 0, image.shape[0] - 1)
                contour_int[:, 1] = np.clip(contour_int[:, 1], 0, image.shape[1] - 1)
                rr, cc = polygon(contour_int[:, 0], contour_int[:, 1], image.shape)
                mask[rr, cc] = True
                
                # Apply erosion with kernel size 1
                from skimage.morphology import binary_erosion, disk
                eroded_mask = binary_erosion(mask, disk(1))
                
                # Convert back to contour
                if np.any(eroded_mask):
                    contours = measure.find_contours(eroded_mask.astype(float), 0.5)
                    if contours:
                        refined_contour = max(contours, key=len)
            except Exception:
                pass  # If erosion fails, use original refined_contour
        
        return refined_contour
    
    def _apply_snake(self, image: np.ndarray, initial_contour: np.ndarray, 
                     params: Dict[str, Any]) -> np.ndarray:
        """Apply snake active contour."""
        try:
            # Smooth the image slightly to help with convergence
            if params.get('pre_smooth', True):
                image = filters.gaussian(image, sigma=params.get('gaussian_sigma', 1.0))
            
            contour = active_contour(
                image,
                initial_contour,
                alpha=params.get('alpha', 0.015),
                beta=params.get('beta', 10),
                gamma=params.get('gamma', 0.001),
                max_px_move=params.get('max_px_move', 1.0),
                max_num_iter=params.get('max_iterations', 2500),
                convergence=params.get('convergence', 0.1)
            )

        except Exception as e:
            warnings.warn(f"Snake active contour failed: {e}")
            return initial_contour

        if params.get('apply_builtin_erosion', True):
            try:
                mask = np.zeros(image.shape, dtype=bool)
                from skimage.draw import polygon
                contour_int = np.round(contour).astype(int)
                contour_int[:, 0] = np.clip(contour_int[:, 0], 0, image.shape[0] - 1)
                contour_int[:, 1] = np.clip(contour_int[:, 1], 0, image.shape[1] - 1)
                rr, cc = polygon(contour_int[:, 0], contour_int[:, 1], image.shape)
                mask[rr, cc] = True
                
                # Apply erosion with kernel size 2 for snake
                from skimage.morphology import binary_erosion, disk
                eroded_mask = binary_erosion(mask, disk(2))
                
                # Convert back to contour
                if np.any(eroded_mask):
                    contours = measure.find_contours(eroded_mask.astype(float), 0.5)
                    if contours:
                        contour = max(contours, key=len)
            except Exception:
                pass  # If erosion fails, use original contour
        
        return contour

    def _apply_graph_cut(self, image: np.ndarray, initial_contour: np.ndarray, 
                        params: Dict[str, Any]) -> np.ndarray:
        """Apply graph cut segmentation."""
        try:
            from skimage.segmentation import random_walker
            
            # Create initial level set from contour
            mask = np.zeros(image.shape, dtype=bool)
            from skimage.draw import polygon
            contour_int = np.round(initial_contour).astype(int)
            contour_int[:, 0] = np.clip(contour_int[:, 0], 0, image.shape[0] - 1)
            contour_int[:, 1] = np.clip(contour_int[:, 1], 0, image.shape[1] - 1)
            rr, cc = polygon(contour_int[:, 0], contour_int[:, 1], image.shape)
            mask[rr, cc] = True
            
            # Create markers (foreground=1, background=2, unknown=0)
            markers = np.zeros(image.shape, dtype=int)
            
            # Mark interior of contour as foreground (shrink slightly to be more conservative)
            from skimage.morphology import binary_erosion, disk
            eroded_mask = binary_erosion(mask, disk(2))
            markers[eroded_mask] = 1
            
            # Create background markers in a ring around the object
            dilated_mask = morphology.binary_dilation(mask, disk(params.get('bg_ring_size', 10)))
            background_ring = dilated_mask & (~mask)
            markers[background_ring] = 2
            
            # Add sparse border markers only if object is not near borders
            border_margin = 20  # pixels from edge
            if (np.any(mask[:border_margin, :]) or np.any(mask[-border_margin:, :]) or 
                np.any(mask[:, :border_margin]) or np.any(mask[:, -border_margin:])):
                # Object is near border, don't add border markers
                pass
            else:
                # Object is away from border, add sparse border markers
                markers[::20, ::20] = 2  # Sparse grid of background markers
            
            # Apply random walker with adjusted parameters
            labels = random_walker(
                image, 
                markers, 
                beta=params.get('beta', 50),  # Lower default beta
                mode=params.get('mode', 'bf')
            )
            
            # Extract contour from result
            result_mask = (labels == 1)
            
            # Apply small morphological cleanup
            result_mask = morphology.binary_opening(result_mask, disk(1))
            
            contours = measure.find_contours(result_mask.astype(float), 0.5)
            if contours:
                return max(contours, key=len)
            else:
                return initial_contour
                
        except Exception as e:
            warnings.warn(f"Graph cut failed: {e}")
            return initial_contour
    
    def _apply_chan_vese(self, image: np.ndarray, initial_contour: np.ndarray, 
                        params: Dict[str, Any]) -> np.ndarray:
        """Apply Chan-Vese active contour."""
        try:
            # Create initial level set from contour
            mask = np.zeros(image.shape, dtype=bool)
            from skimage.draw import polygon
            contour_int = np.round(initial_contour).astype(int)
            contour_int[:, 0] = np.clip(contour_int[:, 0], 0, image.shape[0] - 1)
            contour_int[:, 1] = np.clip(contour_int[:, 1], 0, image.shape[1] - 1)
            rr, cc = polygon(contour_int[:, 0], contour_int[:, 1], image.shape)
            mask[rr, cc] = True
            
            # Apply morphological Chan-Vese
            result = morphological_chan_vese(
                image,
                num_iter=params.get('max_iterations', 35),
                init_level_set=mask,
                smoothing=params.get('smoothing', 3),
                lambda1=params.get('lambda1', 1),
                lambda2=params.get('lambda2', 1)
            )
            
            # Extract contour from result
            contours = measure.find_contours(result.astype(float), 0.5)
            if contours:
                return max(contours, key=len)
            else:
                return initial_contour
                
        except Exception as e:
            warnings.warn(f"Chan-Vese active contour failed: {e}")
            return initial_contour

    
    def apply_refinement_method(self, image: np.ndarray, labels: np.ndarray,
                               method: str = 'snake', params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Apply refinement method to labels on an image slice.
        
        Parameters:
        -----------
        image : np.ndarray
            2D image slice
        labels : np.ndarray
            2D labeled image with initial segmentation
        method : str
            Refinement method ('snake', 'chan_vese', 'geodesic', 'superpixels', 'intensity_snap')
        params : dict
            Parameters for the refinement method
            
        Returns:
        --------
        np.ndarray
            Refined labeled image
        """
        if params is None:
            params = {}
            
        if method not in self.refinement_methods:
            warnings.warn(f"Unknown method {method}, using 'snake'")
            method = 'snake'
        
        # Convert labels to contours (now preserving label IDs)
        contours = self.labels_to_contours(labels)
        
        if not contours:
            return labels
        
        # Apply refinement method to each contour
        refined_contours = []
        for label_id, contour in contours:
            refined_contour = self.refinement_methods[method](image, contour, params)
            refined_contours.append((label_id, refined_contour))
        
        # Convert back to labels (now preserving original IDs)
        refined_labels = self.contours_to_labels(refined_contours, labels.shape)
        
        return refined_labels