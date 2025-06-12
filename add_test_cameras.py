from app import app, db, Camera
from config import DEFAULT_ZONE_POINTS

def add_test_cameras():
    """Add test cameras to the database"""
    with app.app_context():
        # Create test cameras
        test_cameras = [
            {
                'name': 'Front Door',
                'camera_index': 1,
                'is_active': False,
                'zone_points': DEFAULT_ZONE_POINTS
            },
            {
                'name': 'Back Door',
                'camera_index': 0,
                'is_active': True,
                'zone_points': DEFAULT_ZONE_POINTS
            },
            {
                'name': 'Side Entrance',
                'camera_index': 2,
                'is_active': False,
                'zone_points': DEFAULT_ZONE_POINTS
            },
            {
                'name': 'Parking Lot',
                'camera_index': 3,
                'is_active': False,
                'zone_points': DEFAULT_ZONE_POINTS
            }
        ]
        
        # Add cameras to database
        for camera_data in test_cameras:
            camera = Camera(**camera_data)
            db.session.add(camera)
        
        try:
            db.session.commit()
            print("Test cameras added successfully!")
        except Exception as e:
            db.session.rollback()
            print(f"Error adding test cameras: {e}")

if __name__ == '__main__':
    add_test_cameras() 