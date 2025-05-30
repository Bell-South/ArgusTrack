#!/bin/bash
# Optimized GPS-Frequency Processing for Argus Track
# UPDATED: Fixed output paths and added video processing

# Default video file with fallback
VIDEO_FILE="${1:-../fellowship_of_the_frame/data/Videos/Camino_8/FI/GX018691.MP4}"

echo "🚀 Running Argus Track with OPTIMIZED PROCESSING"
echo "==============================================="
echo "📄 Video file: $VIDEO_FILE"
echo "⏰ Started at: $(date)"
echo ""

# Check if video file exists
if [ ! -f "$VIDEO_FILE" ]; then
    echo "❌ Video file not found: $VIDEO_FILE"
    echo "💡 Usage: $0 [path_to_video_file]"
    exit 1
fi

# Get video basename for output files in current directory
VIDEO_BASENAME=$(basename "$VIDEO_FILE" | sed 's/\.[^.]*$//')
echo "📁 Output files will use basename: $VIDEO_BASENAME"

# Check if model exists
MODEL_FILE="../best_2.pt"
if [ ! -f "$MODEL_FILE" ]; then
    echo "⚠️  Model file not found: $MODEL_FILE"
    echo "💡 Please ensure your YOLOv11 model is at $MODEL_FILE"
    echo "   or update the MODEL_FILE variable in this script"
    # Continue with mock detector
    USE_MOCK="true"
else
    USE_MOCK="false"
fi


echo "🚀 FIXED GPS SYNCHRONIZATION MODE"
echo "   🎯 KEY FIX: Only processes frames with actual GPS data"
echo "   📍 No more processing all 7625 frames!"
echo "   🧭 Processes ~10Hz GPS frequency frames only"
echo "   ⚡ Massive performance improvement"
echo ""

echo "✅ OPTIMIZATIONS APPLIED:"
echo "   🎯 GPS-synchronized frame processing (FIXED)"
echo "   📍 Only frames with GPS data are processed"
echo "   🔍 Resolution scaling (50% = 4x fewer pixels to process)"
echo "   🔧 Lower detection threshold: 0.1"
echo "   📊 Detailed performance monitoring"
echo "   🎬 Video output: ENABLED"
echo "   📁 Output location: Current directory (./$VIDEO_BASENAME.*)"
if [ "$USE_MOCK" = "true" ]; then
    echo "   🎭 Using mock detector (model not found)"
fi
echo ""

# Mark start time for timing
START_TIME=$(date +%s)

# Set optimization parameters
RESOLUTION_SCALE="--resolution-scale 0.5"  # Process at half resolution  

echo "🔧 Processing Parameters:"
echo "   📍 GPS Sync: ACTIVE (only GPS frames processed)"
echo "   🔍 Resolution: 0.5x (half size for speed)"
echo "   🎯 Detection threshold: 0.1"
echo "   📁 Output directory: $(pwd)"
echo ""

# Prepare detector arguments
if [ "$USE_MOCK" = "true" ]; then
    DETECTOR_ARGS="--detector mock"
else
    DETECTOR_ARGS="--detector yolov11 --model $MODEL_FILE"
fi

# Output files in current directory
OUTPUT_VIDEO="./${VIDEO_BASENAME}_tracked.mp4"
LOG_FILE="./${VIDEO_BASENAME}_argus.log"

echo "📁 Output files will be:"
echo "   🎬 Video: $OUTPUT_VIDEO"
echo "   📄 JSON: ./${VIDEO_BASENAME}.json"
echo "   🗺️  GeoJSON: ./${VIDEO_BASENAME}.geojson"
echo "   📊 GPS CSV: ./${VIDEO_BASENAME}.csv"
echo "   📋 Log: $LOG_FILE"
echo ""

# Run with GPS synchronization and SAVE VIDEO
echo "🚀 Starting processing..."
argus_track "$VIDEO_FILE" \
    --calibration "$CALIBRATION_FILE" \
    $DETECTOR_ARGS \
    --track-thresh 0.1 \
    --match-thresh 0.3 \
    --track-buffer 50 \
    $RESOLUTION_SCALE \
    --output "$OUTPUT_VIDEO" \
    --log-file "$LOG_FILE" \
    --verbose

EXIT_CODE=$?

# Calculate processing time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "📊 PROCESSING RESULT"
echo "==================="
echo "Exit Code: $EXIT_CODE"
echo "⏱️  Processing completed in ${ELAPSED} seconds"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS!"
    
    # Copy results to current directory with proper names
    echo ""
    echo "📁 Moving output files to current directory..."
    
    # Original output location (same as video)
    VIDEO_DIR=$(dirname "$VIDEO_FILE")
    ORIGINAL_BASENAME=$(basename "$VIDEO_FILE" | sed 's/\.[^.]*$//')
    
    # Copy/move files to current directory
    for ext in json geojson csv; do
        ORIGINAL_FILE="${VIDEO_DIR}/${ORIGINAL_BASENAME}.${ext}"
        TARGET_FILE="./${VIDEO_BASENAME}.${ext}"
        
        if [ -f "$ORIGINAL_FILE" ]; then
            cp "$ORIGINAL_FILE" "$TARGET_FILE"
            echo "   📄 Copied: $TARGET_FILE"
        fi
    done
    
    # Check results
    geojson_file="./${VIDEO_BASENAME}.geojson"
    json_file="./${VIDEO_BASENAME}.json"
    
    if [ -f "$geojson_file" ]; then
        # Try to count geolocated objects (check if jq is available)
        if command -v jq &> /dev/null; then
            geolocated_count=$(jq '.features | length' "$geojson_file" 2>/dev/null || echo "0")
            echo ""
            echo "📍 Geolocated objects: $geolocated_count"
            
            if [ "$geolocated_count" != "0" ] && [ "$geolocated_count" != "null" ]; then
                echo "🎉 SUCCESS: Objects detected and geolocated!"
                
                # Show sample detection
                echo ""
                echo "📋 Sample Detection:"
                jq -r '.features[0] | "   Track \(.properties.track_id) (\(.properties.class_name))\n   Location: \(.geometry.coordinates[1]), \(.geometry.coordinates[0])\n   Confidence: \(.properties.confidence), Reliability: \(.properties.reliability)"' "$geojson_file" 2>/dev/null || echo "   (Unable to parse)"
            else
                echo "⚠️  No geolocated objects found"
                echo ""
                echo "🔍 DEBUGGING INFO:"
                
                # Check if JSON file has tracking data
                if [ -f "$json_file" ]; then
                    echo "   📊 Checking tracking data..."
                    
                    # Check total tracks
                    total_tracks=$(jq '.tracks | length' "$json_file" 2>/dev/null || echo "0")
                    echo "   📈 Total tracks found: $total_tracks"
                    
                    # Check GPS data
                    gps_points=$(jq '.gps_tracks | length' "$json_file" 2>/dev/null || echo "0")
                    echo "   📍 GPS tracks: $gps_points"
                    
                    # Check metadata
                    echo "   📊 Processing metadata:"
                    jq -r '.metadata | "      Processed frames: \(.processed_frames)\n      Skipped frames: \(.skipped_frames)\n      GPS synchronization: \(.gps_synchronization)\n      Geolocated objects: \(.geolocated_objects)"' "$json_file" 2>/dev/null || echo "      (Unable to parse metadata)"
                fi
            fi
        else
            echo "📍 Results saved to: $geojson_file"
            echo "💡 Install jq to see detailed results: sudo apt-get install jq"
        fi
    else
        echo "❌ GeoJSON file not found: $geojson_file"
    fi
    
    # Show output video info
    if [ -f "$OUTPUT_VIDEO" ]; then
        video_size=$(ls -lh "$OUTPUT_VIDEO" | awk '{print $5}')
        echo ""
        echo "🎬 Output video: $OUTPUT_VIDEO ($video_size)"
        echo "💡 View with: ffplay $OUTPUT_VIDEO"
    else
        echo "⚠️  Output video not created: $OUTPUT_VIDEO"
    fi
    
    # Check log file for errors/warnings
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "📋 Log Analysis:"
        echo "   📊 GPS sync statistics:"
        grep -E "GPS points|sync frames|processing ratio" "$LOG_FILE" | tail -3
        echo "   ⚠️  Warnings/Errors:"
        grep -E "WARNING|ERROR" "$LOG_FILE" | tail -5 || echo "      No warnings/errors found"
    fi
    
else
    echo "❌ Failed with exit code $EXIT_CODE"
    
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "🔍 Last few log entries:"
        tail -10 "$LOG_FILE"
    fi
    
    echo ""
    echo "🔍 TROUBLESHOOTING TIPS:"
    echo "1. Check if the video has GPS data:"
    echo "   exiftool \"$VIDEO_FILE\" | grep -i gps"
    echo ""
    echo "2. Try with mock detector to test tracking:"
    echo "   $0 \"$VIDEO_FILE\" --detector mock"
    echo ""
    echo "3. Check video properties:"
    echo "   ffprobe \"$VIDEO_FILE\""
    echo ""
    if [ "$USE_MOCK" = "false" ]; then
        echo "4. Verify your model path:"
        echo "   Make sure $MODEL_FILE exists and is a valid YOLOv11 model"
        echo ""
    fi
    echo "5. Check the log file for detailed error information: $LOG_FILE"
fi

echo ""
echo "💡 FINAL OUTPUT SUMMARY:"
echo "   📁 All files are now in: $(pwd)"
echo "   📄 Results: ./${VIDEO_BASENAME}.json"
echo "   🗺️  Locations: ./${VIDEO_BASENAME}.geojson"
echo "   📊 GPS data: ./${VIDEO_BASENAME}.csv"
echo "   🎬 Video: $OUTPUT_VIDEO"
echo "   📋 Log: $LOG_FILE"
echo ""

if [ $EXIT_CODE -eq 0 ] && [ -f "$geojson_file" ]; then
    geolocated_count=$(jq '.features | length' "$geojson_file" 2>/dev/null || echo "0")
    if [ "$geolocated_count" = "0" ]; then
        echo "🔍 DEBUGGING NEXT STEPS:"
        echo "   1. Check if GPS data was extracted:"
        echo "      head ./${VIDEO_BASENAME}.csv"
        echo ""
        echo "   2. Check detection counts in JSON:"
        echo "      jq '.tracks | length' ./${VIDEO_BASENAME}.json"
        echo ""
        echo "   3. Look for static analysis in log:"
        echo "      grep -i static $LOG_FILE"
        echo ""
        echo "   4. Try with different thresholds:"
        echo "      argus_track \"$VIDEO_FILE\" --track-thresh 0.05 --static-threshold 10.0"
    fi
fi