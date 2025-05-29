#!/bin/bash
# Run with LOWER detection threshold to actually see detections

VIDEO_FILE="${1:-../fellowship_of_the_frame/data/Videos/Camino_8/FI/GX018691.MP4}"

echo "🔧 Running Argus Track with FIXED DETECTION THRESHOLD"
echo "===================================================="
echo "📄 Video file: $VIDEO_FILE"
echo "🔧 Using LOWER threshold to see actual detections"
echo "⏰ Started at: $(date)"
echo ""

echo "🐛 ISSUE IDENTIFIED:"
echo "   ❌ Detection threshold (0.3) was too high"
echo "   ❌ All detections were filtered out (max confidence: 0.14)"
echo "   ❌ Visualization bug with None frame"
echo ""

echo "✅ FIXES APPLIED:"
echo "   🔧 Lower detection threshold: 0.1 (was 0.3)"
echo "   🔧 Fixed visualization None frame bug"
echo "   🔧 Better error handling in real-time display"
echo ""

echo "🎯 Expected Results:"
echo "   ✅ Should now see LED detections (confidence 0.14, 0.06, etc.)"
echo "   ✅ Real-time window should work properly"
echo "   ✅ Track formation should begin"
echo ""

# Run with LOWER threshold to actually detect objects
argus_track "$VIDEO_FILE" \
    --calibration argus_calibration.pkl \
    --detector yolov11 \
    --model ../best_2.pt \
    --track-thresh 0.1 \
    --match-thresh 0.3 \
    --track-buffer 150 \
    --gps-interval 30 \
# --show-realtime \
# --display-size 1280 720 \
    --output tracked_leds_test.mp4 \
    --verbose

EXIT_CODE=$?

echo ""
echo "📊 PROCESSING RESULT"
echo "==================="
echo "Exit Code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS!"
    
    # Check results
    geojson_file="${VIDEO_FILE%.*}.geojson"
    if [ -f "$geojson_file" ]; then
        geolocated_count=$(jq '.features | length' "$geojson_file" 2>/dev/null || echo "0")
        echo "📍 Geolocated objects: $geolocated_count"
        
        if [ "$geolocated_count" != "0" ] && [ "$geolocated_count" != "null" ]; then
            echo "🎉 SUCCESS: Objects detected and geolocated!"
            
            # Show sample detection
            echo ""
            echo "📋 Sample Detection:"
            jq -r '.features[0] | "   Track \(.properties.track_id) (\(.properties.class_name))\n   Location: \(.geometry.coordinates[1]), \(.geometry.coordinates[0])\n   Confidence: \(.properties.confidence), Reliability: \(.properties.reliability)"' "$geojson_file" 2>/dev/null || echo "   (Unable to parse)"
        else
            echo "⚠️  No geolocated objects - but detections should now be visible in real-time"
        fi
    fi
    
elif [ $EXIT_CODE -eq 1 ]; then
    echo "❌ Still failed - checking for specific issues..."
    
    echo ""
    echo "🔍 DEBUGGING STEPS:"
    echo "1. Try without real-time display:"
    echo "   argus_track $VIDEO_FILE --detector yolov11 --model ../best.pt --track-thresh 0.1 --no-realtime --verbose"
    echo ""
    echo "2. Test just detection without tracking:"
    echo "   python3 -c \"
from ultralytics import YOLO
import cv2
model = YOLO('../best.pt')
cap = cv2.VideoCapture('$VIDEO_FILE')
ret, frame = cap.read()
if ret:
    results = model.predict(frame, conf=0.05, verbose=True)
    print(f'Detections: {len(results[0].boxes) if results[0].boxes else 0}')
cap.release()
\""
    echo ""
    echo "3. Check display availability:"
    echo "   echo \$DISPLAY"
    
else
    echo "❌ Failed with exit code $EXIT_CODE"
fi

echo ""
echo "💡 THRESHOLD EXPLANATION:"
echo "   Your model detects objects with confidence 0.06-0.14"
echo "   Previous threshold 0.3 filtered out ALL detections"
echo "   New threshold 0.1 should capture the strongest detections"
echo "   You can adjust further: --track-thresh 0.05 for even more detections"