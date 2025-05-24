import cv2
import numpy as np

class ZoneDetector:
    def __init__(self, initial_points=None):
        """Initialize the zone detector with optional initial points"""
        # Default zone is a rectangle in the middle of the frame
        self.zone_points = initial_points if initial_points else [(100, 100), (400, 100), (400, 300), (100, 300)]
        self.dragging_point = None
        self.point_radius = 5
    
    def is_in_zone(self, bbox):
        """Check if any part of the bounding box overlaps with the zone polygon"""
        x1, y1, x2, y2 = bbox
    
        # Create a polygon from the zone points
        zone_polygon = np.array(self.zone_points, np.int32)
    
        # Check all four corners of the bounding box
        corners = [
            (x1, y1),  # top-left
            (x2, y1),  # top-right
            (x2, y2),  # bottom-right
            (x1, y2)   # bottom-left
        ]
    
        # Check if any corner is inside the polygon
        for corner in corners:
            if cv2.pointPolygonTest(zone_polygon, corner, False) >= 0:
                return True
    
        # Check if the center is inside the polygon
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        if cv2.pointPolygonTest(zone_polygon, (center_x, center_y), False) >= 0:
            return True
    
        # Check if any edge of the bounding box intersects with any edge of the zone polygon
        bbox_edges = [
            [corners[0], corners[1]],  # top edge
            [corners[1], corners[2]],  # right edge
            [corners[2], corners[3]],  # bottom edge
            [corners[3], corners[0]]   # left edge
        ]
    
        zone_edges = []
        for i in range(len(self.zone_points)):
            zone_edges.append([self.zone_points[i], self.zone_points[(i+1) % len(self.zone_points)]])
    
        # Check for line intersections
        for bbox_edge in bbox_edges:
            for zone_edge in zone_edges:
                if self._lines_intersect(bbox_edge[0], bbox_edge[1], zone_edge[0], zone_edge[1]):
                    return True
    
        # Check if zone is completely inside the bounding box
        all_zone_points_inside_bbox = all(
            x1 <= point[0] <= x2 and y1 <= point[1] <= y2 for point in self.zone_points
        )
        if all_zone_points_inside_bbox:
            return True
    
        return False

    def _lines_intersect(self, p1, p2, p3, p4):
        """Check if two line segments intersect"""
        # Convert points to numpy arrays for easier calculation
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        p4 = np.array(p4)
        
        # Calculate the direction vectors
        r = p2 - p1
        s = p4 - p3
        
        # Calculate the cross product
        rxs = np.cross(r, s)
        
        # If cross product is zero, lines are parallel
        if abs(rxs) < 1e-10:
            return False
        
        # Calculate t and u parameters
        q_minus_p = p3 - p1
        t = np.cross(q_minus_p, s) / rxs
        u = np.cross(q_minus_p, r) / rxs
        
        # Check if intersection point lies on both line segments
        return (0 <= t <= 1) and (0 <= u <= 1)
    
    def draw_zone(self, frame):
        """Draw the zone polygon and control points on the frame"""
        # Draw the polygon
        points = np.array(self.zone_points, np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(frame, [points], True, (0, 255, 0), 2)
        
        # Draw the corner points
        for i, point in enumerate(self.zone_points):
            cv2.circle(frame, point, self.point_radius, (0, 255, 0), -1)
            # Add point number label
            cv2.putText(frame, str(i+1), (point[0]-5, point[1]-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def set_zone_points(self, points):
        """Set new zone points"""
        if len(points) == 4:
            self.zone_points = points