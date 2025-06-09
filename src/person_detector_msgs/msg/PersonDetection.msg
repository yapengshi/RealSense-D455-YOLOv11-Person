# person_detector_msgs/msg/PersonDetection.msg
std_msgs/Header header

# 2D Bounding Box in image coordinates (expanded)
int32 nx1
int32 ny1
int32 nx2
int32 ny2

# Estimated 3D position (if valid)
float32 x_3d
float32 y_3d
float32 z_3d
bool is_3d_valid

# Tracking ID (if available)
int32 track_id

# Is this detection considered a real person or a photo?
bool is_real_person # True for real person, False for photo or unsure

# Action string (if available)
string action