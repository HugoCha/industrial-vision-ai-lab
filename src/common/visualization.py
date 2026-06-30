#!/usr/bin/python3

import cv2
import numpy as np

from cv2.typing import MatLike
from typing import Callable, List, Optional

from src.common.file_utils import get_files_by_extension
from src.common.image_loader import ImageLoader
from src.common.image_saver import ImageSaver
from src.common.processors import ProcessKeyResult

def draw_contours(image:MatLike, contours: List[np.ndarray] ) -> MatLike:
    output = image.copy()

    for cnt in contours:
        cv2.drawContours(output, [cnt], -1, (0,255,0), 2)

    return output

def read( path:str ) -> MatLike | None:
    return cv2.imread( path )

def show_image(image:MatLike, title:str="Result"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_images( image_loader:ImageLoader,
                 processed_image_saver:ImageSaver,
                 func:Optional[Callable[[MatLike], MatLike]] = None,
                 process_key:Optional[Callable[[int, MatLike, Optional[MatLike]],None]] = None,
                 can_show_image:bool = False ):
    if ( not image_loader.can_load() ):
        return
    
    img_pathes = get_files_by_extension( image_loader.get_dirpath(), image_loader.get_extension() )
    
    i = 0
    while i < len(img_pathes):
        img_fpath = img_pathes[i]
        loader = ImageLoader.from_filepath( img_fpath )
        if ( loader is not None ):
            result = load_image( loader, processed_image_saver, func, process_key, can_show_image )
            if result == ProcessKeyResult.Next: i+=1
            elif result == ProcessKeyResult.Previous: i = max( 0, i-1 ) 
            elif result == ProcessKeyResult.Quit: break

def load_image( image_loader:ImageLoader,
                processed_image_saver:ImageSaver,
                process_img:Optional[Callable[[MatLike], MatLike]] = None,
                process_key:Optional[Callable[[int, MatLike, Optional[MatLike]],None]] = None,
                can_show_image:bool = False ) -> ProcessKeyResult:
    load = image_loader.load()
    result = ProcessKeyResult.Quit
    if ( not len(load) ):
        return result
    
    img = load[0]
    processed = None

    img_to_process = img
    if can_show_image:
        img_to_process = img.copy()

    if ( process_img is not None ):
        processed = process_img( img_to_process )

    if ( can_show_image ):
        cv2.imshow( "original", img )
        if ( processed is not None ): cv2.imshow( "processed", processed )
        result = _start_process_key_loop( img, processed, process_key )
    elif processed is not None:
        processed_image_saver.save( processed )
    
    cv2.destroyAllWindows()
    return result

    
def capture_image( camera_index:int, 
                   process_img:Optional[Callable[[MatLike], MatLike]] = None,
                   process_key:Optional[Callable[[int, MatLike, Optional[MatLike]],None]] = None ):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        exit()

    ret, frame = cap.read()
    if ret:
        cv2.imshow("capture", frame)

        processed = None
        if ( process_img is not None ):
            processed = process_img( frame.copy() )
            cv2.imshow("processed", frame)

        _start_process_key_loop( frame, processed, process_key )

    cap.release()
    cv2.destroyAllWindows()

def capture_video( camera_index:int, 
                   process_img:Optional[Callable[[MatLike], MatLike]] = None,
                   process_key:Optional[Callable[[int, MatLike, Optional[MatLike]],None]] = None ):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        exit()

    while True:
        ret, frame = cap.read()
        processed = None
        if ret:
            if ( process_img is not None ):
                processed = process_img( frame.copy() )
            else:
                processed = frame

            cv2.imshow( 'camera', processed )
            
            if ( _process_key_loop( frame, processed, process_key ) == ProcessKeyResult.Quit ):
                break

        else:
            print("Error: Could not capture frame.")


    cap.release()
    cv2.destroyAllWindows()

def _process_key_loop( 
        img:MatLike,
        process:Optional[MatLike],
        process_key:Optional[Callable[[int, MatLike, Optional[MatLike]],None]] ) -> ProcessKeyResult:
    key = cv2.waitKey(1)
    
    if key == ord('Q'):
        return ProcessKeyResult.Previous

    if key == ord('S'):
        return ProcessKeyResult.Next

    if key == ord('q'):
        return ProcessKeyResult.Quit

    if key >= 0 and process_key:
        process_key( key, img, process )
    
    return ProcessKeyResult.Continue

def _start_process_key_loop(
        img:MatLike,
        process:Optional[MatLike],
        process_key:Optional[Callable[[int, MatLike, Optional[MatLike]],None]] ) -> ProcessKeyResult:
    
    result = ProcessKeyResult.Continue
    
    while ( result == ProcessKeyResult.Continue ):
        result = _process_key_loop( img, process, process_key )
        continue

    return result