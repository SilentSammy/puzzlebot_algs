import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import NamedTuple

class Detection(NamedTuple):
    obj_pts: np.ndarray  # (N, 3) float32 — 3D object space
    img_pts: np.ndarray  # (N, 2) float32 — 2D image space

def image_to_pdf(img, filepath, physical_width):
    """Convert an image to PDF with exact physical dimensions.
    
    
    Args:
        img: OpenCV image (BGR format) or PIL Image
        filepath: Output PDF file path
        physical_width: Physical width in meters (including margins)
        margin: Margin in meters on each side (default: 0.0)
        
    Returns:
        str: Path to generated PDF file
    """

    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    from PIL import Image
    from reportlab.lib.utils import ImageReader
    
    margin = 0.0 # Deprecated parameter, always set to 0.0

    # Convert to PIL Image if it's an OpenCV image
    if isinstance(img, np.ndarray):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
    else:
        pil_img = img
    
    # Calculate physical height from image aspect ratio
    img_width, img_height = pil_img.size
    aspect_ratio = img_height / img_width
    physical_height = physical_width * aspect_ratio
    
    # Calculate image dimensions (subtract margins from both sides)
    img_physical_width = physical_width - (2 * margin)
    img_physical_height = img_physical_width * aspect_ratio
    
    # Calculate vertical margin to center the image
    # The image maintains its aspect ratio but is smaller, so we need to center it vertically
    vertical_margin = (physical_height - img_physical_height) / 2
    
    # Create PDF canvas sized to exact physical dimensions
    # Convert meters to cm for ReportLab
    width_cm = physical_width * 100
    height_cm = physical_height * 100
    margin_cm = margin * 100
    vertical_margin_cm = vertical_margin * 100
    
    c = canvas.Canvas(filepath, pagesize=(width_cm * cm, height_cm * cm))
    
    # Use ImageReader to wrap PIL image for ReportLab
    img_reader = ImageReader(pil_img)
    
    # Draw image with margins (offset by margin horizontally, centered vertically)
    c.drawImage(img_reader,
               x=margin_cm * cm,
               y=vertical_margin_cm * cm,
               width=img_physical_width * 100 * cm,
               height=img_physical_height * 100 * cm)
    
    c.showPage()
    c.save()
    
    return filepath

class ReferenceDetector(ABC):
    """Base class for all pose estimation reference targets."""
    undistorts = False

    def __init__(self):
        self.world_T: np.ndarray = np.eye(4, dtype=np.float64)

    @abstractmethod
    def detect(self, frame, drawing_frame=None):
        """Detect reference in frame. Returns Detection or None."""
        ...

    @abstractmethod
    def get_board_dimensions(self) -> tuple:
        """Return (width, height) in physical units."""
        ...


class ArucoDetector(ReferenceDetector):
    """Configuration for detecting single ArUco markers."""
    
    def __init__(self, dictionary, marker_id, marker_size, filename=None, K=None, D=None):
        """Initialize ArUco detector for a single marker.
        
        Args:
            dictionary: ArUco dictionary
            marker_id: ID of the marker to detect
            marker_size: Physical size of the marker in meters (edge length)
            filename: Optional base filename for saving/loading (no extension)
        """
        super().__init__()
        self.dictionary = dictionary
        self.marker_id = marker_id
        self.marker_size = marker_size
        self.filename = filename or f"aruco_{marker_id}"
        
        # Create detector
        self.detector = cv2.aruco.ArucoDetector(self.dictionary)
        
        # Generate 3D object points for this marker (corners at Z=0)
        # OpenCV ArUco corners are ordered: top-left, top-right, bottom-right, bottom-left
        half = marker_size / 2.0
        self.obj_points = np.array([
            [-half,  half, 0],  # Top-left
            [ half,  half, 0],  # Top-right
            [ half, -half, 0],  # Bottom-right
            [-half, -half, 0],  # Bottom-left
        ], dtype=np.float32)
        
        # Center point (already centered at origin for single markers)
        self.center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        # Rotation: ArUco doesn't need pose rotation (only img_pts rotation)
        self.needs_rot = False
        self.K = K
        self.D = D

    @property
    def undistorts(self):
        return self.K is not None and self.D is not None

    def detect(self, frame, drawing_frame=None):
        """Detect the specific marker in a frame.
        
        Args:
            frame: Input image/frame to detect marker in
            drawing_frame: Optional frame to draw detected marker on
            
        Returns:
            np.ndarray or None: Marker corners (4x2) if detected, None otherwise
        """
        detect_frame = cv2.undistort(frame, self.K, self.D) if self.undistorts else frame
        corners, ids, _ = self.detector.detectMarkers(detect_frame)
        
        if ids is None:
            return None
        
        # Find our specific marker ID
        ids_flat = ids.flatten()
        mask = ids_flat == self.marker_id
        
        if not mask.any():
            return None
        
        # Get corners for our marker (shape: 1, 4, 2)
        marker_idx = np.where(mask)[0][0]
        marker_corners = corners[marker_idx]
        
        # Draw if requested
        if drawing_frame is not None:
            cv2.aruco.drawDetectedMarkers(drawing_frame, [marker_corners], np.array([[self.marker_id]]))
        
        return Detection(
            obj_pts=self.obj_points.copy(),
            img_pts=marker_corners.reshape(4, 2).astype(np.float32)
        )
    
    def get_board_dimensions(self):
        """Return (width, height) of marker in physical units."""
        return (self.marker_size, self.marker_size)


class QRCodeDetector(ReferenceDetector):
    """Detector for a single QR code by physical size and optional content filter."""

    def __init__(self, qr_size, content=None, K=None, D=None):
        """
        Args:
            qr_size: Physical edge length of the QR code in meters.
            content: Optional decoded string to match. If None, any QR code is accepted.
            K: Optional camera matrix for undistortion before detection.
            D: Optional distortion coefficients for undistortion before detection.
        """
        super().__init__()
        self.qr_size = qr_size
        self.content = content
        self.K = K
        self.D = D
        self.detector = cv2.QRCodeDetector()
        self.needs_rot = False

        half = qr_size / 2.0
        self.obj_points = np.array([
            [-half,  half, 0],  # Top-left
            [ half,  half, 0],  # Top-right
            [ half, -half, 0],  # Bottom-right
            [-half, -half, 0],  # Bottom-left
        ], dtype=np.float32)

    @property
    def undistorts(self):
        # Corners are re-mapped back to the original (distorted) image space in
        # detect(), so the pose pipeline uses the real distortion coefficients.
        return False

    def detect(self, frame, drawing_frame=None):
        use_undistort = self.K is not None and self.D is not None
        detect_frame = cv2.undistort(frame, self.K, self.D) if use_undistort else frame
        try:
            data, points, _ = self.detector.detectAndDecode(detect_frame)
        except cv2.error:
            return None

        if points is None or not data:
            return None

        if self.content is not None and data != self.content:
            return None

        corners = points.reshape(4, 2).astype(np.float32)

        # QR detection ran on the undistorted frame, so corners are in undistorted
        # space. Map them back to the original (distorted) image space so they
        # align with the frame we draw on and so PnP can use the real distortion.
        if use_undistort:
            norm = np.array([[(p[0] - self.K[0, 2]) / self.K[0, 0],
                              (p[1] - self.K[1, 2]) / self.K[1, 1], 1.0]
                             for p in corners], dtype=np.float64)
            redist, _ = cv2.projectPoints(norm, np.zeros(3), np.zeros(3), self.K, self.D)
            corners = redist.reshape(4, 2).astype(np.float32)

        # Roll so index 0 is the corner nearest the image top-left (min x+y),
        # regardless of how the QR code is physically oriented.
        tl_idx = int(np.argmin(corners[:, 0] + corners[:, 1]))
        corners  = np.roll(corners, -tl_idx, axis=0)
        obj_pts = self.obj_points.copy()

        if drawing_frame is not None:
            draw_pts = corners.astype(np.int32)
            cv2.polylines(drawing_frame, [draw_pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            cv2.putText(drawing_frame, data, tuple(draw_pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return Detection(obj_pts=obj_pts.copy(), img_pts=corners)

    def get_board_dimensions(self):
        return (self.qr_size, self.qr_size)


class QReaderDetector(ReferenceDetector):
    """QR code detector using QReader (YOLOv8 + pyzbar).
    More robust than QRCodeDetector at steep angles and difficult lighting.
    Works on the original distorted frame — no undistortion step needed."""

    undistorts = False

    def __init__(self, qr_size, content=None, model_size='s', min_confidence=0.5):
        """
        Args:
            qr_size:        Physical edge length of the QR code in meters.
            content:        Optional decoded string to match (None = accept any).
            model_size:     YOLOv8 model size: 'n', 's', 'm', or 'l'. Default 's'.
            min_confidence: Minimum YOLO detection confidence (0–1). Default 0.5.
        """
        super().__init__()
        self.qr_size = qr_size
        self.content = content
        from qreader import QReader
        self._qreader = QReader(model_size=model_size, min_confidence=min_confidence)
        self.needs_rot = False

        half = qr_size / 2.0
        # Corner order matches cv2.QRCodeDetector: TL, TR, BR, BL
        self.obj_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float32)

    def detect(self, frame, drawing_frame=None):
        detections = self._qreader.detect(image=frame, is_bgr=True)
        if not detections:
            return None

        for det in detections:
            decoded = self._qreader.decode(image=frame, detection_result=det)
            if decoded is None:
                continue
            if self.content is not None and decoded != self.content:
                continue

            # quad_xy: (4, 2) float32 corners in original image space — TL, TR, BR, BL
            corners = np.array(det['quad_xy'], dtype=np.float32).reshape(4, 2)

            if drawing_frame is not None:
                draw_pts = corners.astype(np.int32)
                cv2.polylines(drawing_frame, [draw_pts.reshape(-1, 1, 2)], True, (0, 200, 255), 2)
                cv2.putText(drawing_frame, decoded, tuple(draw_pts[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            return Detection(obj_pts=self.obj_points.copy(), img_pts=corners)

        return None

    def get_board_dimensions(self):
        return (self.qr_size, self.qr_size)


class HybridQRDetector(ReferenceDetector):
    """Fast+accurate hybrid QR detector.

    Every frame:
    - cv2.QRCodeDetector runs synchronously (fast, accurate pose).
    - QReader runs concurrently in a BackgroundPoller thread (slow, robust detection).

    If the primary succeeds its result is returned immediately.
    If the primary fails, the most recent QReader result (from the previous
    background cycle) is returned as a fallback.  The BackgroundPoller ensures
    QReader never blocks the main loop regardless of how often it is called.
    """

    def __init__(self, qr_size, content=None, K=None, D=None,
                 model_size='s', min_confidence=0.5):
        """
        Args:
            qr_size:        Physical edge length of the QR code in meters.
            content:        Optional decoded string to match (None = accept any).
            K, D:           Camera matrix / distortion for the primary detector.
            model_size:     QReader YOLOv8 model size ('n','s','m','l').
            min_confidence: QReader minimum YOLO confidence (0–1).
        """
        super().__init__()
        self.qr_size = qr_size
        self._primary  = QRCodeDetector(qr_size=qr_size, content=content, K=K, D=D)
        self._fallback = QReaderDetector(qr_size=qr_size, content=content,
                                         model_size=model_size,
                                         min_confidence=min_confidence)
        from backg_poller import BackgroundPoller
        self._poller = BackgroundPoller()
        self.needs_rot = False

    @property
    def undistorts(self):
        return self._primary.undistorts

    def detect(self, frame, drawing_frame=None):
        # Submit fallback detection to the background thread.
        # The job draws into its own blank annotation frame so it never races
        # with the main thread's drawing_frame.
        _frame = frame.copy()
        def _fallback_job():
            annot = np.zeros_like(_frame)
            detection = self._fallback.detect(_frame, drawing_frame=annot)
            return detection, annot

        # poll() submits _fallback_job and returns the (detection, annot) pair
        # from the previous background run.
        prev = self._poller.poll(_fallback_job)

        # Run the fast primary synchronously.
        result = self._primary.detect(frame, drawing_frame=drawing_frame)
        if result is not None:
            return result

        # Primary failed — composite the fallback annotation and return its detection.
        if prev is not None:
            fallback_detection, fallback_annot = prev
            if drawing_frame is not None:
                self._poller.composite(drawing_frame, fallback_annot)
            return fallback_detection

        return None

    def get_board_dimensions(self):
        return (self.qr_size, self.qr_size)


class BoardDetector(ReferenceDetector):
    """Base class for board configurations."""
    
    def __init__(self, dictionary, board_width, print_width=None, filename="board"):
        """Initialize board configuration.
        
        Args:
            dictionary: ArUco dictionary
            board_width: Physical width of board content in meters
            print_width: Total width for printing in meters (includes margins). 
                        If None, defaults to board_width (no margins)
            filename: Base filename for saving/loading (no extension)
        """
        super().__init__()
        self.dictionary = dictionary
        self.board_width = board_width
        self.print_width = print_width if print_width is not None else board_width
        self.filename = filename
        self.board: cv2.aruco.Board = self._create_board()
        self.center = self._calculate_center()
        self.board_marker_ids = []  # Override in subclasses
    
    @property
    def image_path(self):
        """Get image filepath with .png extension."""
        return f"{self.filename}.png"
    
    @property
    def pdf_path(self):
        """Get PDF filepath with .pdf extension."""
        return f"{self.filename}.pdf"
    
    def _create_board(self):
        """Create and return the board object. Override in subclasses."""
        pass
    
    def _calculate_center(self):
        """Calculate and return the board's geometric center. Override in subclasses."""
        pass
    
    def detect_corners(self, frame, drawing_frame=None):
        """Detect corners/markers in frame. Override in subclasses."""
        pass
    
    def get_board_dimensions(self):
        """Return (width, height) of board in physical units. Override in subclasses."""
        pass
    
    def get_print_dimensions(self):
        """Return (width, height) for printing including margins.
        
        Returns:
            tuple: (print_width, print_height) in meters
        """
        board_width, board_height = self.get_board_dimensions()
        
        # Calculate margins in meters
        margin_m = (self.print_width - board_width) / 2
        
        # Total print height includes board height plus vertical margins
        print_height = board_height + (2 * margin_m)
        
        return self.print_width, print_height
    
    def generate_image(self, filepath=None, width_px=2160, marker_margin_px=0):
        """Generate board image in pixels with automatic margin calculation.
        
        Args:
            width_px: Width of the board content in pixels (not including margin)
            filepath: Optional path to save the image. If None, image is not saved.
            marker_margin_px: Transparent margin around markers in pixels (default: 4)
            
        Returns:
            Board image with margins based on print_width (BGRA with transparent background)
        """
        # Get board dimensions from subclass
        actual_width, actual_height = self.get_board_dimensions()
        
        # Calculate aspect ratio
        aspect_ratio = actual_height / actual_width
        height_px = int(width_px * aspect_ratio)
        
        # Calculate margin in meters and convert to pixels
        margin_m = (self.print_width - actual_width) / 2
        margin_px = int(margin_m * (width_px / actual_width))

        # Increase generation size so that after margin is applied,
        # the board content is exactly width_px × height_px
        adjusted_width = width_px + (2 * margin_px)
        adjusted_height = height_px + (2 * margin_px)
        
        # Generate board image with margin built-in
        img = self.board.generateImage((adjusted_width, adjusted_height), marginSize=margin_px, borderBits=1)
        
        # Create transparent PNG by masking out the white background
        # Check if image is already grayscale
        if len(img.shape) == 2 or img.shape[2] == 1:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find black pixels (marker borders)
        _, black_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours of marker regions
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create alpha mask by filling all marker contours
        alpha = np.zeros_like(gray)
        cv2.drawContours(alpha, contours, -1, 255, -1)  # Fill all contours
        
        # Optionally dilate to add margin around markers
        if marker_margin_px > 0:
            # Each iteration expands by ~1-2 pixels, so use half the desired margin
            iterations = max(1, marker_margin_px // 2)
            kernel = np.ones((3, 3), np.uint8)
            alpha = cv2.dilate(alpha, kernel, iterations=iterations)
        
        # Add alpha channel to image
        if len(img.shape) == 2:
            # Grayscale image, convert to BGRA
            img_rgba = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        else:
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img_rgba[:, :, 3] = alpha
        img = img_rgba
        
        # Save if filepath provided
        if filepath is not None:
            cv2.imwrite(filepath, img)
        
        return img

    def generate_pdf(self, img, filepath):
        """Generate PDF with board at exact physical dimensions.
        
        Args:
            img: Board image (from generate_image)
            filepath: Output PDF file path
            
        Returns:
            str: Path to generated PDF file
        """
        # Use helper function to create PDF (uses print_width for total page size)
        return image_to_pdf(img, filepath, self.print_width)

class GridboardDetector(BoardDetector):
    """Configuration for ArUco GridBoard with automatic marker separation calculation."""
    def __init__(self, dictionary, size, marker_length, board_width, print_width=None, filename="gridboard"):
        """Initialize GridBoard configuration with automatic marker separation.
        
        Args:
            dictionary: ArUco dictionary
            size: Grid size (cols, rows)
            marker_length: Length of each marker in meters
            board_width: Physical width of board content in meters
            print_width: Total width for printing in meters (includes margins)
            filename: Base filename for saving/loading (no extension)
        """
        self.size = size
        self.marker_length = marker_length
        
        # Calculate marker separation automatically
        # board_width = (n_markers * marker_length) + ((n_markers - 1) * separation)
        # separation = (board_width - n_markers * marker_length) / (n_markers - 1)
        cols, rows = size
        total_marker_width = cols * marker_length
        if cols > 1:
            self.marker_separation = (board_width - total_marker_width) / (cols - 1)
        else:
            self.marker_separation = 0.0
        
        # Call parent constructor
        super().__init__(dictionary, board_width, print_width, filename)
        
        # Rotation: Gridboards need both img_pts and pose rotation
        self.needs_rot = True
        
        # Set board marker IDs
        self.board_marker_ids = list(range(cols * rows))
        
        # Create detector (specific to GridBoard)
        self.detector = cv2.aruco.ArucoDetector(self.dictionary)
    
    def _create_board(self):
        """Create and return the GridBoard."""
        return cv2.aruco.GridBoard(
            size=self.size,
            markerLength=self.marker_length,
            markerSeparation=self.marker_separation,
            dictionary=self.dictionary
        )
    
    def _calculate_center(self):
        """Calculate and return the board's geometric center."""
        cols, rows = self.size
        total_x = (cols * self.marker_length) + ((cols - 1) * self.marker_separation)
        total_y = (rows * self.marker_length) + ((rows - 1) * self.marker_separation)
        return np.array([total_x / 2.0, total_y / 2.0, 0.0], dtype=np.float64)
    
    def get_board_dimensions(self):
        """Return (width, height) of board in physical units."""
        cols, rows = self.size
        actual_width = cols * self.marker_length + (cols - 1) * self.marker_separation
        actual_height = rows * self.marker_length + (rows - 1) * self.marker_separation
        return actual_width, actual_height
    
    def detect_corners(self, frame, drawing_frame=None):
        """Detect markers in a frame and return corners and IDs.
        
        Args:
            frame: Input image/frame to detect markers in
            drawing_frame: Optional frame to draw detected markers on
            
        Returns:
            tuple: (corners, ids) where corners is a list of detected marker corners
                   and ids is an array of corresponding marker IDs
        """
        corners, ids, _ = self.detector.detectMarkers(frame)
        
        if drawing_frame is not None and ids is not None:
            cv2.aruco.drawDetectedMarkers(drawing_frame, corners, ids)
        
        return corners, ids

    def detect(self, frame, drawing_frame=None):
        corners, ids = self.detect_corners(frame, drawing_frame=drawing_frame)
        if ids is None:
            return None
        obj_pts, img_pts = self.board.matchImagePoints(corners, ids)
        if obj_pts is None or obj_pts.shape[0] < 6:
            return None
        obj_pts = obj_pts.reshape(-1, 3).astype(np.float32) - self.center
        img_pts = img_pts.reshape(-1, 2).astype(np.float32)
        return Detection(obj_pts=obj_pts, img_pts=img_pts)

# TODO: Add CharucoBoardConfig similarly if needed

# Instantiate board configurations
board_config_plotter = GridboardDetector(
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    size=(3, 4),
    marker_length=0.05,
    board_width=0.56,
    print_width=0.6,
    filename="resources/gridboard_plotter"
)
board_config_letter = GridboardDetector(
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    size=(3, 4),
    marker_length=0.025,
    board_width=0.15,      # board content width (fits Letter height)
    print_width=0.2159,    # Letter width is 8.5" = 21.59cm
    filename="resources/gridboard_letter"
)
board_config_square = GridboardDetector(
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
    size=(4, 4),
    marker_length=0.05,
    board_width=0.86,
    print_width=0.9,
    filename="resources/gridboard_square"
)
board_config_full = GridboardDetector(
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
    size=(3, 4),
    marker_length=0.1,
    board_width=0.84,
    print_width=0.9,
    filename="resources/gridboard_full"
)

global_detector = board_config_plotter
global_detector = board_config_letter
global_detector = board_config_full
global_detector = board_config_square

if __name__ == "__main__":
    # Example usage: save board image and PDF
    img = global_detector.generate_image(filepath=global_detector.image_path, width_px=2160*4)
    global_detector.generate_pdf(img, global_detector.pdf_path)
    print(global_detector.get_print_dimensions())  # Print dimensions including margins
