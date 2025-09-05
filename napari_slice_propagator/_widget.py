"""
Napari widget for the Slice Propagator plugin.
"""

from typing import TYPE_CHECKING, Optional
import numpy as np
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QDoubleSpinBox, QSpinBox, QPushButton, QComboBox, 
                           QCheckBox, QGroupBox, QFormLayout, QButtonGroup, QRadioButton)
from qtpy.QtCore import Qt
import warnings
import napari

from ._propagator import SlicePropagator

if TYPE_CHECKING:
    import napari

class SlicePropagatorWidget(QWidget):
    """Widget for controlling slice propagation and active contour parameters."""
    
    def __init__(self, napari_viewer: Optional["napari.viewer.Viewer"] = None):
        super().__init__()
        self.viewer = napari_viewer or napari.current_viewer()
        self.propagator = SlicePropagator()
        
        # Keep track of current slice and layer
        self.current_slice = 0
        self.current_labels_layer = None
        self.current_image_layer = None
        self.monitoring_changes = False  # Flag to prevent recursive change detection
        
        # Setup UI
        self.setup_ui()
        
        # Connect to viewer events
        self.viewer.dims.events.current_step.connect(self._on_slice_change)
        self.viewer.layers.events.inserted.connect(self._on_layer_added)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)
        
        # Initialize layer references
        self._update_layer_references()
    
    def setup_ui(self):
        """Create the user interface."""
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Slice Propagator")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; margin: 10px 0;")
        layout.addWidget(title_label)
        
        # Auto-propagation controls
        propagation_group = QGroupBox("Propagation Settings")
        propagation_layout = QFormLayout()
        
        # Radio button group for propagation mode
        self.propagation_mode_group = QButtonGroup()

        self.auto_propagate_off = QRadioButton("No auto-propagation")
        self.auto_propagate_off.setToolTip("Disable automatic propagation when moving through slices")

        self.auto_propagate_basic = QRadioButton("Auto-propagate (basic)")
        self.auto_propagate_basic.setChecked(True)  # Change this to False if you want off by default
        self.auto_propagate_basic.setToolTip("Propagate labels without refinement")

        self.auto_propagate_refined = QRadioButton("Auto-propagate (with refinement)")
        self.auto_propagate_refined.setToolTip("Propagate labels and apply refinement immediately")

        self.propagation_mode_group.addButton(self.auto_propagate_off)
        self.propagation_mode_group.addButton(self.auto_propagate_basic)
        self.propagation_mode_group.addButton(self.auto_propagate_refined)

        propagation_layout.addRow(self.auto_propagate_off)
        propagation_layout.addRow(self.auto_propagate_basic)
        propagation_layout.addRow(self.auto_propagate_refined)

        propagation_group.setLayout(propagation_layout)
        layout.addWidget(propagation_group)
        
        # Auto-mode controls
        auto_mode_group = QGroupBox("Auto-mode Propagation")
        auto_mode_layout = QFormLayout()
        
        # Number of slices to propagate
        self.n_slices_spin = QSpinBox()
        self.n_slices_spin.setRange(1, 50)
        self.n_slices_spin.setValue(5)
        auto_mode_layout.addRow("Number of slices:", self.n_slices_spin)
        
        # Direction selection
        direction_layout = QHBoxLayout()
        self.direction_group = QButtonGroup()
        
        self.forward_radio = QRadioButton("Forward")
        self.backward_radio = QRadioButton("Backward")
        self.forward_radio.setChecked(True)
        
        self.direction_group.addButton(self.forward_radio)
        self.direction_group.addButton(self.backward_radio)
        
        direction_layout.addWidget(self.forward_radio)
        direction_layout.addWidget(self.backward_radio)
        auto_mode_layout.addRow("Direction:", direction_layout)
        
        # Apply refinement checkbox
        self.auto_refinement_check = QCheckBox("Apply refinement during auto-propagation")
        self.auto_refinement_check.setChecked(False)
        auto_mode_layout.addRow(self.auto_refinement_check)

        self.auto_merge_check = QCheckBox("Merge labels after propagation")
        self.auto_merge_check.setChecked(False)
        auto_mode_layout.addRow(self.auto_merge_check)
        
        # Auto-propagate button
        self.auto_propagate_btn = QPushButton("Auto-Propagate")
        self.auto_propagate_btn.clicked.connect(self.auto_propagate)
        auto_mode_layout.addRow(self.auto_propagate_btn)
        
        auto_mode_group.setLayout(auto_mode_layout)
        layout.addWidget(auto_mode_group)
        
        # Label merging controls
        merging_group = QGroupBox("Label Merging")
        merging_layout = QFormLayout()
        
        self.merge_labels_btn = QPushButton("Merge Labels Across Slices")
        self.merge_labels_btn.clicked.connect(self.merge_labels)
        merging_layout.addRow(self.merge_labels_btn)

        self.merge_current_slice_btn = QPushButton("Merge Overlapping Labels (Current Slice)")
        self.merge_current_slice_btn.clicked.connect(self.merge_current_slice)
        merging_layout.addRow(self.merge_current_slice_btn)

        self.overlap_threshold_spin = QDoubleSpinBox()
        self.overlap_threshold_spin.setRange(0.1, 1.0)
        self.overlap_threshold_spin.setValue(0.5)
        self.overlap_threshold_spin.setDecimals(2)
        self.overlap_threshold_spin.setSingleStep(0.05)
        merging_layout.addRow("Overlap threshold:", self.overlap_threshold_spin)
        
        self.similarity_threshold_spin = QDoubleSpinBox()
        self.similarity_threshold_spin.setRange(0.1, 1.0)
        self.similarity_threshold_spin.setValue(0.7)
        self.similarity_threshold_spin.setDecimals(2)
        self.similarity_threshold_spin.setSingleStep(0.05)
        merging_layout.addRow("Similarity threshold:", self.similarity_threshold_spin)
        
        merging_group.setLayout(merging_layout)
        layout.addWidget(merging_group)
        
        # Active contour controls
        contour_group = QGroupBox("Refinement Settings")
        contour_layout = QFormLayout()
        
        # Method selection
        self.method_combo = QComboBox()
        self.method_combo.addItems(['snake', 'chan_vese', 'intensity_snap', 'graph_cut'])
        contour_layout.addRow("Method:", self.method_combo)
        
        # Snake parameters
        self.snake_group = QGroupBox("Snake Parameters")
        snake_layout = QFormLayout()
        
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.001, 1.0)
        self.alpha_spin.setValue(0.015)
        self.alpha_spin.setDecimals(3)
        self.alpha_spin.setSingleStep(0.001)
        snake_layout.addRow("Alpha (continuity):", self.alpha_spin)
        
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(0.1, 100.0)
        self.beta_spin.setValue(10.0)
        self.beta_spin.setDecimals(1)
        snake_layout.addRow("Beta (curvature):", self.beta_spin)
        
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.0001, 0.1)
        self.gamma_spin.setValue(0.001)
        self.gamma_spin.setDecimals(4)
        self.gamma_spin.setSingleStep(0.0001)
        snake_layout.addRow("Gamma (step size):", self.gamma_spin)
        
        self.max_iterations_spin = QSpinBox()
        self.max_iterations_spin.setRange(100, 10000)
        self.max_iterations_spin.setValue(2500)
        snake_layout.addRow("Max iterations:", self.max_iterations_spin)
        
        self.convergence_spin = QDoubleSpinBox()
        self.convergence_spin.setRange(0.01, 1.0)
        self.convergence_spin.setValue(0.1)
        self.convergence_spin.setDecimals(2)
        snake_layout.addRow("Convergence:", self.convergence_spin)
        
        self.snake_group.setLayout(snake_layout)
        contour_layout.addRow(self.snake_group)
        
        # Chan-Vese parameters
        self.cv_group = QGroupBox("Chan-Vese Parameters")
        cv_layout = QFormLayout()
        
        self.cv_iterations_spin = QSpinBox()
        self.cv_iterations_spin.setRange(10, 200)
        self.cv_iterations_spin.setValue(35)
        cv_layout.addRow("Iterations:", self.cv_iterations_spin)
        
        self.smoothing_spin = QSpinBox()
        self.smoothing_spin.setRange(1, 10)
        self.smoothing_spin.setValue(3)
        cv_layout.addRow("Smoothing:", self.smoothing_spin)
        
        self.lambda1_spin = QDoubleSpinBox()
        self.lambda1_spin.setRange(0.1, 10.0)
        self.lambda1_spin.setValue(1.0)
        cv_layout.addRow("Lambda1:", self.lambda1_spin)
        
        self.lambda2_spin = QDoubleSpinBox()
        self.lambda2_spin.setRange(0.1, 10.0)
        self.lambda2_spin.setValue(1.0)
        cv_layout.addRow("Lambda2:", self.lambda2_spin)
        
        self.cv_group.setLayout(cv_layout)
        self.cv_group.setVisible(False)
        contour_layout.addRow(self.cv_group)
        
        # Intensity Snap parameters
        self.snap_group = QGroupBox("Intensity Snap Parameters")
        snap_layout = QFormLayout()
        
        self.search_radius_spin = QSpinBox()
        self.search_radius_spin.setRange(1, 20)
        self.search_radius_spin.setValue(5)
        snap_layout.addRow("Search radius:", self.search_radius_spin)
        
        self.edge_threshold_spin = QDoubleSpinBox()
        self.edge_threshold_spin.setRange(0.01, 1.0)
        self.edge_threshold_spin.setValue(0.1)
        self.edge_threshold_spin.setDecimals(2)
        snap_layout.addRow("Edge threshold:", self.edge_threshold_spin)
        
        self.snap_group.setLayout(snap_layout)
        self.snap_group.setVisible(False)
        contour_layout.addRow(self.snap_group)
        
        contour_group.setLayout(contour_layout)
        layout.addWidget(contour_group)

        # Graph Cut parameters
        self.gc_group = QGroupBox("Graph Cut Parameters")
        gc_layout = QFormLayout()

        self.gc_beta_spin = QSpinBox()
        self.gc_beta_spin.setRange(10, 200)
        self.gc_beta_spin.setValue(50)  # Lower default
        gc_layout.addRow("Beta (edge weight):", self.gc_beta_spin)

        self.gc_ring_size_spin = QSpinBox()
        self.gc_ring_size_spin.setRange(3, 30)
        self.gc_ring_size_spin.setValue(10)
        gc_layout.addRow("Background ring size:", self.gc_ring_size_spin)

        self.gc_group.setLayout(gc_layout)
        self.gc_group.setVisible(False)
        contour_layout.addRow(self.gc_group)

        # Optical Flow parameters group
        self.of_group = QGroupBox("Optical Flow Parameters")
        of_layout = QFormLayout()

        self.enable_optical_flow_check = QCheckBox("Enable optical flow warping")
        self.enable_optical_flow_check.setChecked(True)
        of_layout.addRow(self.enable_optical_flow_check)

        self.of_window_size_spin = QSpinBox()
        self.of_window_size_spin.setRange(5, 50)
        self.of_window_size_spin.setValue(15)
        of_layout.addRow("Window size:", self.of_window_size_spin)

        self.of_group.setLayout(of_layout)
        layout.addWidget(self.of_group)
        
        # Morphological operations group
        morph_group = QGroupBox("Morphological Operations")
        morph_layout = QFormLayout()
        
        # Kernel size for morphological operations
        self.morph_kernel_size_spin = QSpinBox()
        self.morph_kernel_size_spin.setRange(1, 10)
        self.morph_kernel_size_spin.setValue(3)
        morph_layout.addRow("Kernel size:", self.morph_kernel_size_spin)
        
        # Morphological operation buttons
        morph_buttons_layout = QHBoxLayout()
        
        self.dilate_btn = QPushButton("Dilate")
        self.dilate_btn.clicked.connect(lambda: self.apply_morphological_operation('dilation'))
        morph_buttons_layout.addWidget(self.dilate_btn)
        
        self.erode_btn = QPushButton("Erode")
        self.erode_btn.clicked.connect(lambda: self.apply_morphological_operation('erosion'))
        morph_buttons_layout.addWidget(self.erode_btn)
        
        morph_layout.addRow(morph_buttons_layout)
        
        morph_buttons_layout2 = QHBoxLayout()
        
        self.open_btn = QPushButton("Opening")
        self.open_btn.clicked.connect(lambda: self.apply_morphological_operation('opening'))
        morph_buttons_layout2.addWidget(self.open_btn)
        
        self.close_btn = QPushButton("Closing")
        self.close_btn.clicked.connect(lambda: self.apply_morphological_operation('closing'))
        morph_buttons_layout2.addWidget(self.close_btn)
        
        morph_layout.addRow(morph_buttons_layout2)
        
        morph_group.setLayout(morph_layout)
        layout.addWidget(morph_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.apply_refinement_btn = QPushButton("Apply Refinement")
        self.apply_refinement_btn.clicked.connect(self.apply_refinement)
        button_layout.addWidget(self.apply_refinement_btn)
        
        self.manual_propagate_btn = QPushButton("Manual Propagate")
        self.manual_propagate_btn.clicked.connect(self.manual_propagate)
        button_layout.addWidget(self.manual_propagate_btn)
        
        layout.addLayout(button_layout)
        
        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Connect method combo to show/hide parameter groups
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        
    def _on_method_changed(self, method: str):
        """Show/hide parameter groups based on selected method."""
        self.snake_group.setVisible(method == 'snake')
        self.cv_group.setVisible(method == 'chan_vese')
        self.snap_group.setVisible(method == 'intensity_snap')
        self.gc_group.setVisible(method == 'graph_cut')
    
    def _update_layer_references(self):
        """Update references to current image and labels layers."""
        # Find image layer
        image_layers = [layer for layer in self.viewer.layers if hasattr(layer, 'data') and layer.data.ndim == 3]
        if image_layers:
            self.current_image_layer = image_layers[0]
        else:
            self.current_image_layer = None
            
        # Find labels layer
        labels_layers = [layer for layer in self.viewer.layers if layer._type_string == 'labels' and hasattr(layer, 'data') and layer.data.ndim == 3]
        if labels_layers:
            # Disconnect from previous layer if exists
            if self.current_labels_layer is not None:
                self.current_labels_layer.events.data.disconnect(self._on_labels_changed)
            
            self.current_labels_layer = labels_layers[0]
            # Connect to new layer
            self.current_labels_layer.events.data.connect(self._on_labels_changed)
        else:
            self.current_labels_layer = None

        self._set_image_data_for_propagator() #This is new for optical flow step
    
    def _on_layer_added(self, event):
        """Handle new layer addition."""
        self._update_layer_references()
    
    def _on_layer_removed(self, event):
        """Handle layer removal."""
        self._update_layer_references()
    
    def _on_labels_changed(self, event):
        """Handle changes to the labels layer data."""
        if self.monitoring_changes or self.current_labels_layer is None:
            return
            
        # Detect manual changes on current slice
        current_slice = self.current_slice
        current_labels = self.current_labels_layer.data[current_slice]
        
        # Only detect changes if we're not in the middle of auto-propagation
        if not self.monitoring_changes:
            self.propagator.detect_manual_changes(current_slice, current_labels)
    
    def _on_slice_change(self, event):
        """Handle slice change events with incremental propagation."""

        # Check if auto-propagation is disabled
        if self.auto_propagate_off.isChecked():
            self.current_slice = self.viewer.dims.current_step[0]  # Still update current slice
            return
            
        if not (self.auto_propagate_basic.isChecked() or self.auto_propagate_refined.isChecked()):
            return
            
        # Get current slice index (assuming first dimension is the slice dimension)
        new_slice = self.viewer.dims.current_step[0]
        
        if new_slice == self.current_slice:
            return
            
        previous_slice = self.current_slice
        self.current_slice = new_slice
        
        # Only propagate if we have both image and labels layers
        if self.current_image_layer is None or self.current_labels_layer is None:
            self._update_layer_references()
            if self.current_image_layer is None or self.current_labels_layer is None:
                return
        
        try:
            self._auto_propagate_incremental(previous_slice, new_slice)
        except Exception as e:
            warnings.warn(f"Auto-propagation failed: {e}")
            self.status_label.setText(f"Error: {str(e)}")
    

    def _auto_propagate_incremental(self, from_slice: int, to_slice: int):
        """
        Automatically propagate new annotations using incremental propagation.
        This allows adding labels to slices that already have annotations.
        """
        labels_data = self.current_labels_layer.data
        
        # Check if source slice has annotations
        if not self.propagator.has_annotations(labels_data[from_slice]):
            return

        self._set_image_data_for_propagator()
        
        # Use incremental propagation method
        self.monitoring_changes = True
        try:
            # Skip postprocessing if refinement will be applied
            apply_postprocessing = not self.auto_propagate_refined.isChecked()
            
            updated_labels = self.propagator.incremental_propagate(
                from_slice, to_slice, labels_data, apply_postprocessing=apply_postprocessing
            )

            # Apply refinement if auto-propagate with refinement is enabled
            if self.auto_propagate_refined.isChecked() and self.current_image_layer is not None:
                method = self.method_combo.currentText()
                params = self.get_current_parameters()
                
                target_labels = updated_labels[to_slice]
                refined_labels = self.propagator.apply_refinement_method(
                    self.current_image_layer.data[to_slice], target_labels, method, params
                )
                
                # Apply postprocessing AFTER refinement
                refined_labels = self.propagator.apply_postprocessing(refined_labels)
                updated_labels[to_slice] = refined_labels
            
            # Update the labels layer only if propagation actually happened
            if not np.array_equal(updated_labels, labels_data):
                self.current_labels_layer.data = updated_labels
                self.current_labels_layer.refresh()
                
                # Count how many new labels were propagated
                new_labels_on_target = set(np.unique(updated_labels[to_slice])) - set(np.unique(labels_data[to_slice]))
                new_labels_on_target.discard(0)
                
                if new_labels_on_target:
                    refinement_text = f" with {method}" if self.auto_propagate_refined.isChecked() else ""
                    self.status_label.setText(
                        f"Incrementally propagated {len(new_labels_on_target)} new label(s) from slice {from_slice} to {to_slice}{refinement_text}"
                    )
                else:
                    self.status_label.setText("No new labels to propagate")
            else:
                self.status_label.setText("No new labels to propagate")
                    
        finally:
            self.monitoring_changes = False

    
    def manual_propagate(self):
        """Manually propagate current slice to next slice."""
        if self.current_labels_layer is None:
            self.status_label.setText("No labels layer found")
            return
            
        current_slice = self.current_slice
        next_slice = current_slice + 1
        
        if next_slice >= self.current_labels_layer.data.shape[0]:
            self.status_label.setText("Already at last slice")
            return
            
        try:
            self._auto_propagate_incremental(current_slice, next_slice)
            # Move to next slice
            new_step = list(self.viewer.dims.current_step)
            new_step[0] = next_slice
            self.viewer.dims.current_step = new_step
        except Exception as e:
            self.status_label.setText(f"Manual propagation failed: {str(e)}")
    
    def auto_propagate(self):
        """Auto-propagate labels from current slice to multiple slices."""
        if self.current_labels_layer is None:
            self.status_label.setText("No labels layer found")
            return
        
        if self.current_image_layer is None and self.auto_refinement_check.isChecked():
            self.status_label.setText("Need image layer for refinement")
            return
        
        current_slice = self.current_slice
        n_slices = self.n_slices_spin.value()
        direction = 'forward' if self.forward_radio.isChecked() else 'backward'
        apply_refinement = self.auto_refinement_check.isChecked()
        
        try:
            self.status_label.setText(f"Auto-propagating to {n_slices} slices ({direction})...")
            self.monitoring_changes = True
            
            # Get refinement parameters if needed
            refinement_params = None
            refinement_method = None
            if apply_refinement:
                refinement_method = self.method_combo.currentText()
                refinement_params = self.get_current_parameters()
            
            # Apply batch propagation
            updated_data = self.propagator.batch_propagate(
                current_slice,
                self.current_labels_layer.data,
                n_slices,
                direction=direction,
                apply_refinement=apply_refinement,
                refinement_method=refinement_method,
                refinement_params=refinement_params,
                image_data=self.current_image_layer.data if self.current_image_layer else None
            )
            
            # Update the labels layer
            self.current_labels_layer.data = updated_data
            self.current_labels_layer.refresh()

            if self.auto_merge_check.isChecked():
                self.status_label.setText("Merging labels...")
                similarity_threshold = self.similarity_threshold_spin.value()
                merged_data = self.propagator.merge_labels_across_slices(
                    updated_data, similarity_threshold=similarity_threshold
                )
                self.current_labels_layer.data = merged_data
                self.current_labels_layer.refresh()
                
                # Update all snapshots since IDs have changed
                for slice_idx in range(merged_data.shape[0]):
                    self.propagator.slice_snapshots[slice_idx] = merged_data[slice_idx].copy()

            
            refinement_text = f" with {refinement_method} refinement" if apply_refinement else ""
            merge_text = " and merged" if self.auto_merge_check.isChecked() else ""
            self.status_label.setText(f"Auto-propagated to {n_slices} slices ({direction}){refinement_text}{merge_text}")
        
        except Exception as e:
            self.status_label.setText(f"Auto-propagation failed: {str(e)}")
            warnings.warn(f"Auto-propagation failed: {e}")
        finally:
            self.monitoring_changes = False
    
    def merge_labels(self):
        """Merge labels across slices so same objects have same IDs."""
        if self.current_labels_layer is None:
            self.status_label.setText("No labels layer found")
            return
        
        try:
            self.status_label.setText("Merging labels across slices...")
            self.monitoring_changes = True
            
            similarity_threshold = self.similarity_threshold_spin.value()
            
            # Apply label merging
            merged_data = self.propagator.merge_labels_across_slices(
                self.current_labels_layer.data, 
                similarity_threshold=similarity_threshold
            )
            
            # Update the labels layer
            self.current_labels_layer.data = merged_data
            self.current_labels_layer.refresh()
            
            # Update all snapshots since IDs have changed
            for slice_idx in range(merged_data.shape[0]):
                self.propagator.slice_snapshots[slice_idx] = merged_data[slice_idx].copy()
            
            self.status_label.setText(f"Successfully merged labels across slices (threshold: {similarity_threshold})")
            
        except Exception as e:
            self.status_label.setText(f"Label merging failed: {str(e)}")
            warnings.warn(f"Label merging failed: {e}")
        finally:
            self.monitoring_changes = False

    def merge_current_slice(self):
        """Merge overlapping labels on current slice."""
        if self.current_labels_layer is None:
            self.status_label.setText("No labels layer found")
            return
    
        try:
            current_slice = self.current_slice
            labels_slice = self.current_labels_layer.data[current_slice]
            
            if not self.propagator.has_annotations(labels_slice):
                self.status_label.setText("No annotations on current slice")
            
            self.status_label.setText("Merging overlapping labels on current slice...")
            self.monitoring_changes = True
            
            overlap_threshold = self.overlap_threshold_spin.value()
            
            # Apply intra-slice merging
            merged_slice = self.propagator.merge_overlapping_labels_within_slice(
                labels_slice, overlap_threshold=overlap_threshold
            )
            
            # Update the labels layer
            updated_data = self.current_labels_layer.data.copy()
            updated_data[current_slice] = merged_slice
            self.current_labels_layer.data = updated_data
            self.current_labels_layer.refresh()
            
            # Update snapshot
            self.propagator.slice_snapshots[current_slice] = merged_slice.copy()
            
            self.status_label.setText(f"Merged overlapping labels on slice {current_slice}")
            
        except Exception as e:
            self.status_label.setText(f"Intra-slice merging failed: {str(e)}")
            warnings.warn(f"Intra-slice merging failed: {e}")
        finally:
            self.monitoring_changes = False
    
    def apply_morphological_operation(self, operation: str):
        """Apply morphological operations to current slice."""
        if self.current_labels_layer is None:
            self.status_label.setText("No labels layer found")
            return
        
        try:
            current_slice = self.current_slice
            labels_slice = self.current_labels_layer.data[current_slice]
            
            if not self.propagator.has_annotations(labels_slice):
                self.status_label.setText("No annotations on current slice")
                return
            
            kernel_size = self.morph_kernel_size_spin.value()
            
            self.monitoring_changes = True
            
            # Apply morphological operation
            modified_labels = self.propagator.apply_morphological_operation(
                labels_slice, operation, kernel_size
            )
            
            # Update the labels layer
            updated_data = self.current_labels_layer.data.copy()
            updated_data[current_slice] = modified_labels
            self.current_labels_layer.data = updated_data
            self.current_labels_layer.refresh()
            
            # Update metadata - mark as manual since user initiated
            modified_unique = set(np.unique(modified_labels))
            modified_unique.discard(0)
            if current_slice in self.propagator.label_metadata:
                self.propagator.label_metadata[current_slice]['manual'].update(modified_unique)
                self.propagator.label_metadata[current_slice]['auto'].difference_update(modified_unique)
                self.propagator.label_metadata[current_slice]['all_existing'] = modified_unique.copy()
            
            # Update snapshot
            self.propagator.slice_snapshots[current_slice] = modified_labels.copy()
            
            self.status_label.setText(f"Applied {operation} with kernel size {kernel_size}")
            
        except Exception as e:
            self.status_label.setText(f"Morphological operation failed: {str(e)}")
            warnings.warn(f"Morphological operation failed: {e}")
        finally:
            self.monitoring_changes = False

    def _set_image_data_for_propagator(self):
        """Pass current image data to propagator for optical flow."""
        if self.current_image_layer is not None:
            self.propagator._current_image_data = self.current_image_layer.data
        else:
            self.propagator._current_image_data = None
    
    def get_current_parameters(self) -> dict:
        """Get current refinement parameters based on selected method."""
        method = self.method_combo.currentText()
        
        if method == 'snake':
            return {
                'alpha': self.alpha_spin.value(),
                'beta': self.beta_spin.value(),
                'gamma': self.gamma_spin.value(),
                'max_iterations': self.max_iterations_spin.value(),
                'convergence': self.convergence_spin.value()
            }

        elif method == 'chan_vese':
            return {
                'max_iterations': self.cv_iterations_spin.value(),
                'smoothing': self.smoothing_spin.value(),
                'lambda1': self.lambda1_spin.value(),
                'lambda2': self.lambda2_spin.value()
            }
            
        elif method == 'intensity_snap':
            return {
                'search_radius': self.search_radius_spin.value(),
                'edge_threshold': self.edge_threshold_spin.value()
            }

        elif method == 'graph_cut':
            return {
                'beta': self.gc_beta_spin.value(),
                'bg_ring_size': self.gc_ring_size_spin.value(),
                'mode': 'bf'
            }
        
        return {}
        
    # This function is below what claude predicted. This is from a previous version of the file.
    def apply_refinement(self):
        """Apply refinement method to current slice."""
        if self.current_image_layer is None or self.current_labels_layer is None:
            self.status_label.setText("Need both image and labels layers")
            return
        
        try:
            # Get current slice data
            current_slice = self.current_slice
            image_slice = self.current_image_layer.data[current_slice]
            labels_slice = self.current_labels_layer.data[current_slice]
            
            if not self.propagator.has_annotations(labels_slice):
                self.status_label.setText("No annotations on current slice")
                return
            
            # Get parameters and method
            method = self.method_combo.currentText()
            params = self.get_current_parameters()
            
            # Apply refinement
            self.status_label.setText(f"Applying {method}...")
            
            self.monitoring_changes = True  # Prevent change detection during manual refinement
            
            try:
                refined_labels = self.propagator.apply_refinement_method(
                    image_slice, labels_slice, method, params
                )
                
                # Update the labels layer
                updated_data = self.current_labels_layer.data.copy()
                updated_data[current_slice] = refined_labels
                self.current_labels_layer.data = updated_data
                self.current_labels_layer.refresh()
                
                # Mark refined labels as manual (since user initiated this)
                refined_unique = set(np.unique(refined_labels))
                refined_unique.discard(0)
                self.propagator.label_metadata[current_slice]['manual'].update(refined_unique)
                self.propagator.label_metadata[current_slice]['auto'].difference_update(refined_unique)
                self.propagator.label_metadata[current_slice]['all_existing'] = refined_unique.copy()
                
                # Update snapshot
                self.propagator.slice_snapshots[current_slice] = refined_labels.copy()
                
                self.status_label.setText(f"Applied {method} refinement to {len(refined_unique)} label(s)")
                
            finally:
                self.monitoring_changes = False
                
        except Exception as e:
            self.status_label.setText(f"Refinement failed: {str(e)}")
            warnings.warn(f"Refinement application failed: {e}")
            self.monitoring_changes = False


# For napari plugin discovery
def make_slice_propagator_widget():
    """Factory function to create the widget."""
    # Get the current napari viewer instance
    import napari
    viewer = napari.current_viewer()
    return SlicePropagatorWidget(viewer)