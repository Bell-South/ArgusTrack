#!/bin/bash
# Installation script for Argus Track

echo "🚀 Installing Argus Track System"
echo "================================"

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "📋 Python version: $python_version"

if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
    echo "❌ Python 3.8+ required, found $python_version"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🏗️  Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install the package in development mode
echo "📦 Installing Argus Track in development mode..."
pip install -e .

# Install extra dependencies
echo "📦 Installing extra dependencies..."
pip install -e ".[visualization,gps,stereo]"

# Create calibration file
echo "🔧 Creating sample calibration file..."
python3 -c "
import sys
sys.path.insert(0, '.')
exec(open('create_calibration.py').read())
"

# Verify installation
echo ""
echo "✅ Verifying installation..."
if command -v argus_track &> /dev/null; then
    echo "✅ argus_track command available"
    argus_track --help | head -10
else
    echo "❌ argus_track command not found"
    echo "💡 Try running: source venv/bin/activate"
fi

# Check for YOLOv11 model
if [ ! -f "../best_2.pt" ]; then
    echo ""
    echo "⚠️  Model file not found: ../best_2.pt"
    echo "💡 Please ensure your YOLOv11 model is available at ../best_2.pt"
    echo "   or update the model path in run_argus.sh"
fi

# Check for video file
video_file="../fellowship_of_the_frame/data/Videos/Camino_8/FI/GX018691.MP4"
if [ ! -f "$video_file" ]; then
    echo ""
    echo "⚠️  Test video not found: $video_file"
    echo "💡 Please ensure your test video is available or update the path in run_argus.sh"
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate the environment: source venv/bin/activate"
echo "   2. Test with: ./run_argus.sh [video_path]"
echo "   3. Or run directly: argus_track your_video.mp4 --detector yolov11 --model your_model.pt"