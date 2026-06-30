#!/usr/bin/python3

import argparse

from cv2.typing import MatLike

from src.common.image_loader import ImageLoader, ImageLoaderParameters
from src.common.image_saver import ImageSaverParameters
from src.common.launcher import Launcher, LauncherParameters, LaunchOption
from src.common.processors import DefaultKeysProcessor, ImageProcessor, KeyProcessor, KeysProcessor

from .camera_calibration import CameraCalibrationParameters, CameraCalibration
from .config import *

class ManualCameraCalibrationProcessor(ImageProcessor, DefaultKeysProcessor):
    def __init__(self, 
                 parameters:CameraCalibrationParameters,
                 img_saver_params:ImageSaverParameters, 
                 process_img_saver_params:ImageSaverParameters ):
        DefaultKeysProcessor.__init__( self, img_saver_params, process_img_saver_params )

        self.parameters = parameters
        self.sub_menus().update( {
            'p': KeyProcessor( 'p', "Display calibration parameters", lambda im, proc: print( self.parameters ) ),
            '+': KeyProcessor( '+', "Add image to calibration images", lambda im, proc: self.add( im )  ),
            '-': KeyProcessor( '-', "Remove last image from calibration images", lambda im, proc: self.remove()  ),
            'c': KeyProcessor( 'c', "Clear calibration images", lambda im, proc: self.clear()  ),
            's': KeyProcessor( 's', f"Add and Save image to file in {self.parameters.chessboard_path}", lambda im, proc: self.save( im ) )
        } )

    def add( self, img:MatLike ):
        self.parameters.chessboard_images.append( img )
        print( f"Image added, image cnt: {len(self.parameters.chessboard_images)}")

    def remove( self ):
        if ( self.parameters.chessboard_image_count() > 0 ):
            self.parameters.chessboard_images.pop()
            print( f"Image remove, image cnt: {self.parameters.chessboard_image_count()}")

    def clear( self ):
        self.parameters.chessboard_images.clear()
        print( f"Chessboard images cleared" )

    def save( self, img:MatLike ):
        if self.raw_image_saver.can_save() and self.raw_image_saver.save( img ):
            self.add( img )

    def title(self) -> str:
        return "Camera calibration"

    def quit_menu(self) -> str:
        return f"Quit and Save calibration file in {self.parameters.calibration_path}"

    def process_img(self, img:MatLike ) -> MatLike:
        ret = CameraCalibration.find_chessboard_corners( img, self.parameters )
        if ( ret is not None ):
            return ret[1]
        return img

def parse_args():
    parser = argparse.ArgumentParser(description="Process chessboard images.")

    # camera index
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help=f"Camera index. (default: {CAMERA_INDEX})"
    )

    # live calibration
    parser.add_argument(
        "--live",
        type=bool,
        default=LIVE_CALIBRATION,
        help=f"Is inlive calibration. (default: {LIVE_CALIBRATION})"
    )

    # chessboard folder path
    parser.add_argument(
        "--chessboard-path",
        type=str,
        default=CHESSBOARD_PATH,
        help=f"Folder path to chessboard files (default: {CHESSBOARD_PATH})"
    )

    # chessboard dimensions
    parser.add_argument(
        "--chessboard",
        type=int,
        nargs=2, 
        default=CHESSBOARD,
        metavar=("WIDTH", "HEIGHT"),
        help=f"Optional chessboard dimensions as (WIDTH, HEIGHT) (default: {CHESSBOARD})"
    )

    # output path
    parser.add_argument(
        "--output-path",
        type=str,
        default=CAMERA_PATH,
        help=f"Folder path to save the camera calibration file. (default: {CAMERA_PATH})"
    )

    args = parser.parse_args()

    chessboard = None
    if args.chessboard:
        chessboard = tuple(args.chessboard)

    return args.chessboard_path, args.camera, args.live, chessboard, args.output_path

if __name__ == "__main__":

    chessboard_path, camera_index, live, chessboard, output_path = parse_args()

    parameters = CameraCalibrationParameters( chessboard, chessboard_path, output_path )
    print( f"Calibration parameters:\n{parameters}")
    
    img_loader_params = ImageLoaderParameters( chessboard_path, IMAGE_EXTENSION )
    img_saver_params = ImageSaverParameters( chessboard_path, IMAGE_BASE_NAME, IMAGE_EXTENSION, False )
    process_img_saver_params = ImageSaverParameters( DATA_PROCESSED_PATH, IMAGE_BASE_NAME, IMAGE_EXTENSION, False )
    
    if ( live and camera_index >= 0 ):
        processor = ManualCameraCalibrationProcessor( 
            parameters, 
            img_saver_params, 
            process_img_saver_params )
    
        launcher_params = LauncherParameters( 
            img_loader_params,
            img_saver_params,
            process_img_saver_params,
            camera_index, 
            LaunchOption.CAPTURE_VIDEO )
    
        launcher = Launcher( launcher_params, processor, processor )
        launcher.launch()
    else:
        img_loader = ImageLoader( img_loader_params )
        parameters.chessboard_images = img_loader.load()
    
    try:
        CameraCalibration.calibrate( parameters, True )
    except:
        print( "Camera calibration failed" )