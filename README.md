# Napari Slice Propagator

A Napari plugin for semi-automated 3D segmentation using slice propagation and active contours.

## Features

- **Automatic slice propagation**: Annotations automatically propagate to neighboring slices when you navigate through a 3D stack
- **Multiple active contour methods**: Snake, Chan-Vese, and Geodesic active contours
- **Smart propagation**: Only propagates to slices that don't already have annotations
- **Bidirectional support**: Works when moving forward or backward through slices
- **Parameter control**: Adjustable parameters for fine-tuning active contour behavior
- **Manual override**: Option to manually propagate and apply active contours

## Installation

### Method 1: Development Installation (Recommended)

1. **Create the plugin directory structure** on your computer:
   ```
   napari-slice-propagator/
   ├── napari_slice_propagator/
   │   ├── __init__.py
   │   ├── _widget.py
   │   ├── _propagator.py
   │   └── napari.yaml
   ├── setup.cfg
   └── README.md
   ```

2. **Copy the provided files** into the appropriate locations in this directory structure.

3. **Install in development mode** from your conda environment:
   ```bash
   # Navigate to the plugin directory
   cd napari-slice-propagator
   
   # Install in development mode
   pip install -e .
   ```

### Method 2: Direct Installation
```bash
# If you have the plugin as a package
pip install napari-slice-propagator
```

## Usage

### Basic Workflow

1. **Launch Napari** with your conda environment activated:
   ```bash
   napari
   ```

2. **Load a 3D image stack** into Napari (File → Open...)

3. **Create a Labels layer** (Layer → New Labels Layer) or convert an existing layer to labels

4. **Open the plugin**:
   - Go to Plugins → Slice Propagator
   - The plugin widget will appear on the right side of the interface

5. **Start annotating**:
   - Use Napari's built-in drawing tools in the Labels layer to annotate objects on one slice
   - Navigate to the next slice - your annotations will automatically propagate!
   - Make manual adjustments as needed
   - Use "Apply Active Contour" to refine the propagated annotations

### Plugin Controls

#### Propagation Settings
- **Auto-propagate annotations**: When checked, annotations automatically propagate when you change slices (default: enabled)

#### Active Contour Methods

**Snake Active Contour** (default):
- **Alpha (continuity)**: Controls contour smoothness (0.001-1.0, default: 0.015)
- **Beta (curvature)**: Controls resistance to bending (0.1-100, default: 10.0)
- **Gamma (step size)**: Controls convergence speed (0.0001-0.1, default: 0.001)
- **Max iterations**: Maximum number of iterations (100-10000, default: 2500)
- **Convergence**: Convergence threshold (0.01-1.0, default: 0.1)

**Chan-Vese Active Contour**:
- **Iterations**: Number of iterations (10-200, default: 35)
- **Smoothing**: Smoothing factor (1-10, default: 3)
- **Lambda1/Lambda2**: Weight parameters (0.1-10.0, default: 1.0)

**Geodesic Active Contour**:
- **Iterations**: Number of iterations (50-500, default: 230)
- **Smoothing**: Smoothing factor (1-5, default: 1)
- **Balloon force**: Expansion/contraction force (-1.0 to 1.0, default: 0.0)

#### Manual Controls
- **Apply Active Contour**: Manually apply active contour to current slice
- **Manual Propagate**: Manually propagate current slice to next slice

### Tips for Best Results

1. **Start with clear, well-defined objects** in your first slice
2. **Use the snake method** for most applications - it's the most stable
3. **Adjust alpha and beta** based on your object characteristics:
   - Lower alpha for more flexible contours
   - Higher beta for smoother contours
4. **Fine-tune after active contour** using Napari's manual editing tools
5. **The plugin won't overwrite existing annotations** - so you can always make manual corrections

### Troubleshooting

**Plugin doesn't appear in menu**:
- Make sure you installed the plugin in the same conda environment as Napari
- Try restarting Napari
- Check that all dependencies are installed

**Auto-propagation not working**:
- Ensure you have both an image layer and a labels layer
- Check that "Auto-propagate annotations" is enabled
- Make sure your image and labels layers are 3D

**Active contour not improving results**:
- Try different methods (snake usually works best)
- Adjust parameters - start with default values and make small changes
- Ensure your initial annotation roughly outlines the object

**Memory issues with large stacks**:
- The plugin processes one slice at a time, so memory usage should be minimal
- If you encounter issues, try closing other applications or working with smaller image stacks

## Requirements

- Python >= 3.8
- napari[all]
- numpy
- scikit-image
- scipy

## Contributing

This plugin was developed to accelerate 3D annotation workflows by leveraging the redundancy between neighboring slices. Feel free to suggest improvements or report issues!

## License

BSD-3-Clause